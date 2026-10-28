"""Hybrid RAG retrieval and embedding service for Ren.

Provides:
- Paragraph and sentence-aware chunking (target 500 words, range 400-700, 60-word overlap)
- Anchor preservation (PDF pages, Markdown headings, section ordinals)
- Deterministic chunk content hashing (SHA-256)
- OpenAI-compatible batch embeddings (max 32 chunks/request) with cost control
- Hybrid retrieval combining FTS5 top 4 + semantic top 4, deduplicated by chunk_id,
  exact keyword match boosting, returning up to 6 final chunks
- Grounded context formatting [S1] filename — anchor and structured citations
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "from", "have", "into",
    "more", "notes", "that", "this", "with", "your", "the", "for", "how", "what",
    "with", "which", "will", "would", "there", "their", "they", "were", "when",
}


def now() -> str:
    return datetime.now(UTC).isoformat()


def compute_content_hash(text: str) -> str:
    """Deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting paragraph breaks."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    paragraphs = re.split(r"\n{2,}", text)
    sentences: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        raw_sents = re.split(r"(?<=[.!?])\s+", para)
        for s in raw_sents:
            s = s.strip()
            if s:
                sentences.append(s)
    return sentences


def chunk_text(
    text: str,
    target_words: int = 500,
    min_words: int = 400,
    max_words: int = 700,
    overlap_words: int = 60,
) -> list[str]:
    """Split text by paragraphs/sentences into chunks of 400-700 words (target 500).

    Adjacent chunks share exactly `overlap_words` (default 60) words from the end
    of the previous chunk.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []

    total_words = sum(len(s.split()) for s in sentences)
    if total_words <= max_words:
        return [" ".join(sentences)]

    chunks: list[str] = []
    idx = 0
    overlap_prefix = ""

    while idx < len(sentences):
        current_sents: list[str] = []
        prefix_words = overlap_prefix.split() if overlap_prefix else []
        current_words = len(prefix_words)

        # Check if all remaining sentences can be absorbed into this final chunk
        remaining_words = sum(len(s.split()) for s in sentences[idx:])
        if current_words + remaining_words <= max_words and chunks:
            current_sents = sentences[idx:]
            chunk_str = (overlap_prefix + " " + " ".join(current_sents)).strip()
            chunks.append(chunk_str)
            break

        while idx < len(sentences):
            sent = sentences[idx]
            sent_w = len(sent.split())

            # If a single sentence exceeds max_words, handle it gracefully
            if sent_w > max_words and current_words == 0:
                words = sent.split()
                # Take target_words slice
                take = min(target_words, len(words))
                current_sents.append(" ".join(words[:take]))
                current_words += take
                # Replace remaining sentence in sentences list
                remaining_sentence = " ".join(words[take:])
                sentences[idx] = remaining_sentence
                break

            if current_words + sent_w <= max_words:
                if current_words < min_words:
                    current_sents.append(sent)
                    current_words += sent_w
                    idx += 1
                elif current_words + sent_w <= target_words:
                    current_sents.append(sent)
                    current_words += sent_w
                    idx += 1
                else:
                    # Compare distance to target
                    if abs(current_words + sent_w - target_words) < abs(current_words - target_words):
                        current_sents.append(sent)
                        current_words += sent_w
                        idx += 1
                    break
            else:
                if current_words >= min_words:
                    break
                else:
                    # Current words is below min_words, but adding sent exceeds max_words
                    current_sents.append(sent)
                    current_words += sent_w
                    idx += 1
                    break

        chunk_str = (overlap_prefix + " " + " ".join(current_sents)).strip()
        chunks.append(chunk_str)

        # Set overlap prefix for the next chunk
        words = chunk_str.split()
        if len(words) >= overlap_words:
            overlap_prefix = " ".join(words[-overlap_words:])
        else:
            overlap_prefix = chunk_str

    return chunks


def extract_sections_with_anchors(
    path: Path | None,
    text: str,
    source_kind: str = "text",
) -> list[tuple[str, str]]:
    """Preserve page or section anchors where available."""
    # 1. PDF files: extract per page if file exists
    if path and path.suffix.lower() == ".pdf" and path.exists():
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            sections: list[tuple[str, str]] = []
            for page_num, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    sections.append((f"Page {page_num}", page_text))
            if sections:
                return sections
        except Exception:
            pass

    # 2. Markdown headings detection in text
    matches = list(re.finditer(r"^(#{1,6}\s+.+)$", text, flags=re.MULTILINE))
    if matches:
        sections = []
        if matches[0].start() > 0:
            pre = text[:matches[0].start()].strip()
            if pre:
                sections.append(("Section 1", pre))
        for i, m in enumerate(matches):
            heading = re.sub(r"^#{1,6}\s+", "", m.group(1)).strip(" #:-")
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            if body:
                sections.append((heading, body))
        if sections:
            return sections

    # 3. Default fallback: single section
    return [("Section 1", text)]


def chunk_source_document(
    path: Path | None,
    text: str,
    source_kind: str = "text",
    target_words: int = 500,
    min_words: int = 400,
    max_words: int = 700,
    overlap_words: int = 60,
) -> list[dict[str, Any]]:
    """Produce structured chunks with source_id (set by caller), ordinal, text, anchor, and content_hash."""
    sections = extract_sections_with_anchors(path, text, source_kind)
    all_chunks: list[dict[str, Any]] = []
    ordinal = 1

    for section_anchor, section_text in sections:
        section_chunks = chunk_text(
            section_text,
            target_words=target_words,
            min_words=min_words,
            max_words=max_words,
            overlap_words=overlap_words,
        )
        for chunk_str in section_chunks:
            if not chunk_str.strip():
                continue
            all_chunks.append({
                "ordinal": ordinal,
                "text": chunk_str,
                "anchor": section_anchor if len(sections) > 1 or section_anchor != "Section 1" else f"Chunk {ordinal}",
                "content_hash": compute_content_hash(chunk_str),
            })
            ordinal += 1

    return all_chunks


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm1) * math.sqrt(norm2))


def call_openai_embeddings_api(
    base_url: str,
    api_key: str,
    model: str,
    texts: list[str],
    timeout: int = 40,
) -> list[list[float]]:
    """Call OpenAI/OpenRouter-compatible embeddings endpoint."""
    payload = json.dumps({
        "model": model,
        "input": texts,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    data = res.get("data", [])
    data.sort(key=lambda item: item.get("index", 0))
    return [item["embedding"] for item in data]


def exact_keyword_match_boost(query: str, text: str) -> float:
    """Slightly boost chunks with exact keyword matches when scores are close."""
    q = query.lower().strip()
    t = text.lower()
    boost = 0.0
    # Exact full query match
    if len(q) >= 3 and q in t:
        boost += 0.08
    # Exact word/term match (including hyphenated terms like "tf-idf")
    words = [w for w in re.findall(r"[\w-]{2,}", q) if w not in STOP_WORDS]
    if words:
        matches = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", t))
        boost += 0.04 * (matches / len(words))
    return boost


def build_grounded_context_and_citations(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Format the grounded chat context and structured citations for up to 6 chunks.

    Block format:
    [S1] filename — page/section/anchor
    chunk text
    """
    final_chunks = chunks[:6]
    context_blocks: list[str] = []
    citations: list[dict[str, Any]] = []

    for idx, c in enumerate(final_chunks, start=1):
        filename = c.get("filename") or "Untitled"
        anchor = c.get("anchor") or f"Chunk {idx}"
        text = c.get("text", "").strip()

        context_blocks.append(f"[S{idx}] {filename} — {anchor}\n{text}")
        citations.append({
            "source_id": c.get("source_id", ""),
            "filename": filename,
            "anchor": anchor,
            "chunk_id": c.get("chunk_id") or c.get("id", ""),
        })

    return "\n\n".join(context_blocks), citations


def embed_chunks_for_source(
    db: Any,
    chunks: list[dict[str, Any]],
    known_vectors: dict[str, tuple[int, str]] | None,
    base_url: str,
    api_key: str | None,
    model: str,
) -> tuple[int, int, str | None]:
    """Cost control: create embeddings only for newly imported or changed chunk hashes.

    Never re-embed unchanged chunks.
    Batch embedding requests where supported, maximum 32 chunks/request.
    Returns: (newly_embedded_count, reused_count, failure_or_skip_message)
    """
    if not chunks:
        return 0, 0, None

    if known_vectors is None:
        known_vectors = {}
        all_hashes = list({c["content_hash"] for c in chunks})
        placeholders = ",".join("?" for _ in all_hashes)
        rows = db.execute(
            f"SELECT content_hash, dimensions, vector_json FROM embeddings WHERE content_hash IN ({placeholders}) AND model=?",
            all_hashes + [model],
        ).fetchall()
        for r in rows:
            known_vectors[r["content_hash"]] = (r["dimensions"], r["vector_json"])

    reused_count = 0
    to_embed: list[dict[str, Any]] = []

    for c in chunks:
        h = c["content_hash"]
        if h in known_vectors:
            dims, vec_json = known_vectors[h]
            db.execute(
                """INSERT INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     model=excluded.model, dimensions=excluded.dimensions,
                     vector_json=excluded.vector_json, content_hash=excluded.content_hash""",
                (str(uuid.uuid4()), c["chunk_id"], model, dims, vec_json, h, now()),
            )
            reused_count += 1
        else:
            to_embed.append(c)

    if not to_embed:
        return 0, reused_count, None

    if not api_key:
        return 0, reused_count, "Embeddings skipped: no API key configured"

    # Group by unique hash so identical chunks in the same source are embedded once
    hash_to_text: dict[str, str] = {}
    for c in to_embed:
        if c["content_hash"] not in hash_to_text:
            hash_to_text[c["content_hash"]] = c["text"]

    hash_items = list(hash_to_text.items())
    newly_embedded = 0
    batch_size = 32

    for i in range(0, len(hash_items), batch_size):
        batch = hash_items[i : i + batch_size]
        batch_hashes = [h for h, _ in batch]
        batch_texts = [txt for _, txt in batch]

        try:
            vectors = call_openai_embeddings_api(base_url, api_key, model, batch_texts)
        except Exception as err:
            return newly_embedded, reused_count, f"Embedding generation failed: {err}"

        for h, vec in zip(batch_hashes, vectors):
            dims = len(vec)
            vec_json = json.dumps(vec)
            known_vectors[h] = (dims, vec_json)
            for c in to_embed:
                if c["content_hash"] == h:
                    db.execute(
                        """INSERT INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(chunk_id) DO UPDATE SET
                             model=excluded.model, dimensions=excluded.dimensions,
                             vector_json=excluded.vector_json, content_hash=excluded.content_hash""",
                        (str(uuid.uuid4()), c["chunk_id"], model, dims, vec_json, h, now()),
                    )
                    newly_embedded += 1

    return newly_embedded, reused_count, None


def query_fts_chunks(
    db: Any,
    query: str,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Retrieve top chunks from SQLite FTS5."""
    raw_terms = re.findall(r"[\w-]{2,}", query)
    if not raw_terms:
        return []
    filtered = [term for term in raw_terms if term.lower() not in STOP_WORDS]
    terms = filtered if filtered else raw_terms
    match = " OR ".join(f'"{term}"' for term in terms[:12])
    try:
        rows = db.execute(
            """SELECT c.id AS chunk_id, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename,
                      bm25(chunks_fts) AS bm25_score
               FROM chunks_fts f
               JOIN chunks c ON c.id=f.chunk_id
               JOIN sources s ON s.id=c.source_id
               WHERE chunks_fts MATCH ?
               ORDER BY bm25(chunks_fts) LIMIT ?""",
            (match, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        rows = db.execute(
            """SELECT c.id AS chunk_id, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename, 0.0 AS bm25_score
               FROM chunks c
               JOIN sources s ON s.id=c.source_id
               WHERE c.text LIKE ? LIMIT ?""",
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def retrieve_hybrid_chunks(
    db: Any,
    query: str,
    base_url: str,
    api_key: str | None,
    embedding_model: str,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Hybrid retrieval:
    1. Keep FTS5 retrieval first (retrieve top 4).
    2. If embeddings are available:
       - Embed only the user query.
       - Compare query embedding with stored vectors using cosine similarity.
       - Retrieve top 4 semantic chunks.
    3. Combine FTS5 top 4 + semantic top 4.
    4. Deduplicate by chunk_id.
    5. Rank exact keyword matches slightly higher when scores are close.
    6. Return a maximum of 6 final chunks.
    """
    fts_top4 = query_fts_chunks(db, query, limit=4)

    stored_vectors: list[Any] = []
    if api_key:
        try:
            stored_vectors = db.execute(
                """SELECT e.chunk_id, e.vector_json, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename
                   FROM embeddings e
                   JOIN chunks c ON c.id=e.chunk_id
                   JOIN sources s ON s.id=c.source_id
                   WHERE e.model=?""",
                (embedding_model,),
            ).fetchall()
        except Exception:
            stored_vectors = []

    if not api_key or not stored_vectors:
        # FTS5 only fallback (if query has more FTS5 matches, can return up to limit)
        if len(fts_top4) < limit:
            more_fts = query_fts_chunks(db, query, limit=limit)
            return more_fts if more_fts else fts_top4
        return fts_top4

    # Embed only the user query
    try:
        query_vectors = call_openai_embeddings_api(base_url, api_key, embedding_model, [query])
        query_vec = query_vectors[0] if query_vectors else None
    except Exception:
        query_vec = None

    if not query_vec:
        if len(fts_top4) < limit:
            more_fts = query_fts_chunks(db, query, limit=limit)
            return more_fts if more_fts else fts_top4
        return fts_top4

    # Semantic similarity scoring
    semantic_scored: list[tuple[float, dict[str, Any]]] = []
    for row in stored_vectors:
        try:
            vec = json.loads(row["vector_json"])
            sim = cosine_similarity(query_vec, vec)
            item = dict(row)
            item["chunk_id"] = row["chunk_id"]
            semantic_scored.append((sim, item))
        except Exception:
            continue

    semantic_scored.sort(key=lambda x: x[0], reverse=True)
    semantic_top4 = semantic_scored[:4]

    # Combine FTS5 top 4 + semantic top 4, deduplicate by chunk_id
    candidates: dict[str, dict[str, Any]] = {}

    fts_rank_scores = [0.85, 0.75, 0.65, 0.55]
    for rank, fts_chunk in enumerate(fts_top4):
        cid = fts_chunk["chunk_id"]
        candidates[cid] = {
            "chunk": fts_chunk,
            "fts_score": fts_rank_scores[rank] if rank < len(fts_rank_scores) else 0.50,
            "sem_score": 0.0,
            "in_fts": True,
            "in_sem": False,
        }

    for sem_score, sem_chunk in semantic_top4:
        cid = sem_chunk["chunk_id"]
        if cid in candidates:
            candidates[cid]["sem_score"] = max(0.0, sem_score)
            candidates[cid]["in_sem"] = True
        else:
            candidates[cid] = {
                "chunk": sem_chunk,
                "fts_score": 0.0,
                "sem_score": max(0.0, sem_score),
                "in_fts": False,
                "in_sem": True,
            }

    scored_candidates: list[tuple[float, dict[str, Any]]] = []
    for cid, data in candidates.items():
        chunk = data["chunk"]
        sem_score = data["sem_score"]
        fts_score = data["fts_score"]

        if data["in_fts"] and data["in_sem"]:
            base_score = 0.5 * sem_score + 0.5 * fts_score + 0.15
        elif data["in_sem"]:
            base_score = sem_score
        else:
            base_score = fts_score

        # Exact keyword boost when scores are close
        boost = exact_keyword_match_boost(query, chunk.get("text", ""))
        final_score = base_score + boost
        scored_candidates.append((final_score, chunk))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    final_chunks = [c for _, c in scored_candidates[:limit]]
    return final_chunks

