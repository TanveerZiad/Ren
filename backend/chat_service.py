"""Provider-neutral, source-grounded chat completion for Ren.

Supports:
1. Local CPU Extractive Synthesis (`provider == 'local_cpu'` or fallback when no external API is reachable/configured)
   - Runs 100% locally on CPU with zero GPU or API key required
   - Synthesizes answers directly from retrieved Ren source excerpts with [S1], [S2] inline citations
2. Local CPU LLM Servers (`ollama`, `lmstudio`, `llamacpp`) via OpenAI-compatible localhost endpoints
3. Cloud OpenAI-compatible providers (`openrouter`, `openai`)
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from backend.rag_service import STOP_WORDS


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    provider_used: bool
    error: str | None = None
    mode: str = "local_cpu"


def grounded_system_prompt(context: str, memories: list[str]) -> str:
    memory_text = "\n".join(f"- {memory}" for memory in memories[:8]) or "- No saved personal memory yet."
    return f"""You are Ren, a thoughtful personal knowledge companion.
Answer only from the provided Ren source excerpts.
If the evidence is insufficient, say so.
Cite every important claim as [S1], [S2], etc.
Do not use outside knowledge unless web search is explicitly enabled.
Keep answers direct, honest, and practical.

Saved user memory:
{memory_text}

Relevant Ren source excerpts:
{context or 'No relevant source excerpts were found.'}"""


def synthesize_local_cpu_answer(question: str, context: str) -> str:
    """Synthesize a grounded, cited answer purely on the local CPU from retrieved context blocks.

    Parses `[S1] filename — anchor\\ntext` blocks, scores sentences against the user's
    question, and formats a concise grounded response with `[S1]`, `[S2]` citations.
    """
    if not context or "not enough evidence for a reliable answer" in context.lower():
        return "I found some loosely related material, but not enough evidence for a reliable answer."

    blocks = re.split(r"\n\n(?=\[S\d+\]\s)", context.strip())
    parsed_blocks: list[tuple[str, str, str]] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if not lines:
            continue
        header = lines[0].strip()
        m = re.match(r"^(\[S\d+\])\s+(.+)$", header)
        cite_tag = m.group(1) if m else "[S1]"
        source_label = m.group(2) if m else header
        body = " ".join(line.strip() for line in lines[1:] if line.strip())
        if body:
            parsed_blocks.append((cite_tag, source_label, body))

    if not parsed_blocks:
        return "I found some loosely related material, but not enough evidence for a reliable answer."

    q_words = [w for w in re.findall(r"[\w-]{2,}", question.lower()) if w not in STOP_WORDS]

    scored_sentences: list[tuple[float, int, str, str]] = []
    seen_norm: set[str] = set()

    for block_idx, (cite_tag, _label, body) in enumerate(parsed_blocks):
        raw_sents = re.split(r"(?<=[.!?])\s+|\n+", body)
        for sent_idx, sent in enumerate(raw_sents):
            sent_clean = re.sub(r"\s+", " ", sent).strip()
            if len(sent_clean) < 18:
                continue
            norm = sent_clean.lower()
            if norm in seen_norm:
                continue
            seen_norm.add(norm)

            s_tokens = set(re.findall(r"[\w-]{2,}", norm))
            if q_words:
                exact_hits = sum(1 for w in q_words if w in s_tokens or ( "-" in w and w in norm))
                stem_hits = sum(
                    0.5 for w in q_words
                    if w not in s_tokens and len(w) >= 5 and any(t.startswith(w[:5]) for t in s_tokens if len(t) >= 5)
                )
                relevance = (exact_hits + stem_hits) / len(q_words)
            else:
                relevance = 0.5

            # Prefer earlier blocks (higher reranker score) and definition-like sentences
            rank_bonus = max(0.0, 0.35 - (block_idx * 0.08) - (sent_idx * 0.02))
            def_bonus = 0.12 if re.search(r"\b(is|are|means|refers|combines|maintains|uses|provides|keeps|stores)\b", norm) else 0.0
            total_score = relevance + rank_bonus + def_bonus
            scored_sentences.append((total_score, block_idx, cite_tag, sent_clean))

    if not scored_sentences:
        first_tag, _, first_body = parsed_blocks[0]
        snippet = first_body[:360].rstrip(" .,;") + "."
        return f"{snippet} {first_tag}"

    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    top_selected = scored_sentences[:3]
    # Order selected sentences by source block order for natural reading flow
    top_selected.sort(key=lambda x: x[1])

    synthesized_parts: list[str] = []
    for _score, _b_idx, cite_tag, sent_text in top_selected:
        if not sent_text.endswith((".", "!", "?")):
            sent_text += "."
        synthesized_parts.append(f"{sent_text} {cite_tag}")

    return " ".join(synthesized_parts)


def complete_chat(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    conversation: list[dict[str, str]],
    context: str,
    memories: list[str],
    thinking: str = "balanced",
    provider: str = "local_cpu",
) -> ChatCompletion:
    is_local_cpu_mode = (
        provider == "local_cpu"
        or model == "local-cpu-extractive"
        or (not api_key and provider not in {"ollama", "lmstudio", "llamacpp"} and "localhost" not in base_url and "127.0.0.1" not in base_url)
    )

    if is_local_cpu_mode:
        user_question = conversation[-1]["content"] if conversation else ""
        answer = synthesize_local_cpu_answer(user_question, context)
        return ChatCompletion(
            content=answer,
            provider_used=False,
            error=None,
            mode="local_cpu",
        )

    effective_key = api_key or "local-cpu"
    temperature = {"focused": 0.1, "balanced": 0.3, "creative": 0.7}.get(thinking, 0.3)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": grounded_system_prompt(context, memories)}, *conversation[-16:]],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {effective_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        return ChatCompletion(content=content, provider_used=True, error=None, mode=provider)
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        # Graceful CPU fallback: never block the user from reading their own local knowledge
        user_question = conversation[-1]["content"] if conversation else ""
        fallback_answer = synthesize_local_cpu_answer(user_question, context)
        return ChatCompletion(
            content=fallback_answer,
            provider_used=False,
            error=None,
            mode="local_cpu_fallback",
        )
