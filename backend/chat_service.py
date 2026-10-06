"""Provider-neutral, source-grounded chat completion for Ren.

The local API owns persistence. This module only turns a selected model, a
conversation, and the user's retrieved Ren context into a provider request.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatCompletion:
    content: str
    provider_used: bool
    error: str | None = None


def grounded_system_prompt(context: str, memories: list[str]) -> str:
    memory_text = "\n".join(f"- {memory}" for memory in memories[:8]) or "- No saved personal memory yet."
    return f"""You are Ren, a thoughtful personal knowledge companion.
Use the supplied Ren context as evidence. Never claim a source says something it does not.
If context is insufficient, say what is missing. Keep answers direct and practical.
When using a source excerpt, cite it inline as [S1], [S2], etc.

Saved user memory:
{memory_text}

Relevant Ren source excerpts:
{context or 'No relevant source excerpts were found.'}"""


def complete_chat(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    conversation: list[dict[str, str]],
    context: str,
    memories: list[str],
    thinking: str = "balanced",
) -> ChatCompletion:
    if not api_key:
        return ChatCompletion(
            content="No AI provider is configured yet. Add an OpenRouter-compatible API key in Settings to use Ren Chat. Your imported knowledge remains local.",
            provider_used=False,
            error="provider_not_configured",
        )

    temperature = {"focused": 0.1, "balanced": 0.3, "creative": 0.7}.get(thinking, 0.3)
    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": grounded_system_prompt(context, memories)}, *conversation[-16:]],
        "temperature": temperature,
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result["choices"][0]["message"]["content"].strip()
        return ChatCompletion(content=content, provider_used=True)
    except urllib.error.HTTPError as error:
        return ChatCompletion(content="Ren could not reach the configured AI provider. Check the API key, model name, and provider URL.", provider_used=False, error=f"provider_http_{error.code}")
    except (urllib.error.URLError, KeyError, IndexError, json.JSONDecodeError):
        return ChatCompletion(content="Ren could not reach the configured AI provider. Your message was saved locally; try again after checking the connection.", provider_used=False, error="provider_unavailable")
