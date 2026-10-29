"""Hybrid RAG retrieval and embedding service for Ren.

Provides:
- Paragraph and sentence-aware chunking (target 500 words, range 400-700, 60-word overlap)
- Anchor preservation (PDF pages, Markdown headings, section ordinals)
- Deterministic chunk content hashing (SHA-256)
- OpenAI-compatible batch embeddings (max 32 chunks/request) with cost control and local fallback
- 2-Stage Hybrid retrieval:
  Stage 1: Broad recall (10-15 candidates from FTS5 + semantic embeddings)
  Stage 2: Lightweight multi-signal reranker (0.55 semantic + 0.35 keyword/proximity + 0.10 title/source,
           dynamically favoring keyword relevance on exact concepts like 'TF-IDF')
  Stage 3: Confidence threshold gating (CONFIDENCE_THRESHOLD = 0.30) and 3-pass source/section diversity
- Metadata filters: source file, topic, recent imports, and knowledge cards only
- Grounded context formatting [S1] filename — anchor and structured citations with excerpts
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

STOP_WORDS = {
    "about", "after", "again", "also", "and", "are", "been", "from", "have", "into",
    "more", "notes", "that", "this", "with", "your", "the", "for", "how", "what",
    "which", "will", "would", "there", "their", "they", "were", "when", "does", "do",
    "can", "could", "should", "where", "who", "whom", "whose", "why", "use", "used",
    "using", "make", "makes", "made", "help", "helps", "know", "tell", "explain",
    "describe", "between", "under", "over", "some", "such", "than", "then", "them",
    "these", "those", "very", "just", "like", "only", "other", "into", "onto", "upon",
    "was", "is", "in", "on", "of", "to", "at", "by", "an", "as", "be", "or", "if",
    "it", "its", "you", "my", "me", "we", "our", "us", "has", "had", "not", "no",
}

SYNONYM_GROUPS = [
    {"revise", "revision", "revisit", "review", "remember", "resurface", "spaced", "repetition", "retain", "memory", "queue", "cards"},
    {"tfidf", "tf-idf", "term", "frequency", "inverse", "document", "idf", "bm25", "lexical", "keyword", "weights", "rare"},
    {"semantic", "meaning", "paraphrase", "paraphrases", "vector", "vectors", "embedding", "embeddings", "cosine", "similarity", "dense"},
    {"hybrid", "combine", "combines", "combining", "both", "lexical", "semantic", "precision", "recall"},
    {"local", "locally", "privacy", "private", "offline", "sqlite", "disk", "directory", "appdata", "cloud"},
    {"organize", "organization", "mess", "inbox", "folder", "folders", "manual", "rule", "automatic"},
    {"format", "formats", "file", "files", "pdf", "txt", "markdown", "accept", "accepts", "import", "imported", "drop"},
    {"original", "copy", "managed", "safe", "untouched", "preserve", "preserved"},
    {"raft", "consensus", "leader", "follower", "candidate", "election", "replication", "paxos", "state", "states", "log"},
    {"cap", "theorem", "consistency", "availability", "partition", "partitions", "network", "trade-off", "tradeoff"},
    {"clock", "clocks", "causality", "causal", "partial", "ordering", "concurrent", "vector"},
]

CONFIDENCE_THRESHOLD = 0.30


def now() -> str:
    return datetime.now(UTC).isoformat()


def compute_content_hash(text: str) -> str:
    """Deterministic SHA-256 hash of normalized text."""
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def generate_local_embedding(text: str, dim: int = 256) -> list[float]:
    """Deterministic local semantic embedding using signed random projections (SimHash-style)
    over tokens, stems, and concept-synonym groups.

    Signed projections (+1 / -1) ensure unrelated documents have near-zero expected cosine
    similarity instead of positive bucket-collision bias.
    """
    vec = [0.0] * dim
    words = [w for w in re.findall(r"[\w-]{2,}", text.lower()) if w not in STOP_WORDS]
    if not words:
        return vec

    def _add_feature(feature_key: str, weight: float) -> None:
        digest = hashlib.md5(feature_key.encode("utf-8")).digest()
        # Project each feature across 4 signed dimensions to suppress single-bucket collisions
        for i in range(4):
            idx = ((digest[i * 2] << 8) | digest[i * 2 + 1]) % dim
            sign = 1.0 if (digest[8 + i] & 1) == 0 else -1.0
            vec[idx] += sign * weight

    for w in words:
        # 1. Exact token projection
        _add_feature(f"tok:{w}", 1.0)

        # 2. Stem projection (first 5 chars) to align morphological variants (revise / revision)
        stem = w.replace("-", "")[:5]
        if len(stem) >= 4:
            _add_feature(f"stem:{stem}", 0.55)

        # 3. Concept group projection to align paraphrases
        for grp_idx, grp in enumerate(SYNONYM_GROUPS):
            if w in grp or any(len(w) >= 5 and w.startswith(g[:5]) for g in grp if len(g) >= 5):
                _add_feature(f"concept:{grp_idx}", 0.85)

    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0.0 else vec


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
                take = min(target_words, len(words))
                current_sents.append(" ".join(words[:take]))
                current_words += take
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
                    if abs(current_words + sent_w - target_words) < abs(current_words - target_words):
                        current_sents.append(sent)
                        current_words += sent_w
                        idx += 1
                    break
            else:
                if current_words >= min_words:
                    break
                else:
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
    """Boost chunks with exact keyword or technical term matches."""
    q = query.lower().strip()
    t = text.lower()
    boost = 0.0
    if len(q) >= 3 and q in t:
        boost += 0.08
    words = [w for w in re.findall(r"[\w-]{2,}", q) if w not in STOP_WORDS]
    if words:
        matches = sum(1 for w in words if re.search(r"\b" + re.escape(w) + r"\b", t))
        boost += 0.06 * (matches / len(words))
    return boost


def has_exact_concept_token(query: str) -> bool:
    """Detect whether the query asks about an exact acronym, hyphenated term, or quoted phrase (e.g. 'TF-IDF', 'BM25', 'CAP')."""
    if re.search(r'"[^"]+"', query):
        return True
    for token in re.findall(r"\b[\w-]{2,}\b", query):
        # Uppercase acronym (2+ uppercase letters/digits like TF, IDF, BM25, CAP, SQL)
        if re.match(r"^[A-Z0-9-]{2,}$", token):
            return True
        # Hyphenated technical term like TF-IDF
        if "-" in token and len(token) >= 4:
            return True
    return False


def extract_query_filters(query: str) -> tuple[str, dict[str, Any]]:
    """Extract metadata filters (source file, topic, recency, knowledge cards only) from natural query language."""
    filters: dict[str, Any] = {}
    cleaned = query.strip()

    # Pattern: in/from filename.ext or source:filename.ext
    m_file = re.search(r"\b(?:in|from|source:)\s+([a-zA-Z0-9_\-\.]+\.(?:txt|md|markdown|pdf))\b", cleaned, re.I)
    if m_file:
        filters["source_filename"] = m_file.group(1).strip()
        cleaned = cleaned[:m_file.start()] + " " + cleaned[m_file.end():]

    # Pattern: topic:topic_name or in topic "topic_name"
    m_topic = re.search(
        r"\b(?:topic:([a-zA-Z0-9_\-]+)|in\s+topic\s+[\"']([^\"']+)[\"']|in\s+topic\s+([a-zA-Z0-9_\-]+))",
        cleaned,
        re.I,
    )
    if m_topic:
        topic_val = (m_topic.group(1) or m_topic.group(2) or m_topic.group(3) or "").strip()
        if topic_val:
            filters["topic_name"] = topic_val
        cleaned = cleaned[:m_topic.start()] + " " + cleaned[m_topic.end():]

    # Pattern: recent:N or in the last N days
    m_recent = re.search(r"\b(?:recent:(\d+)|in the last (\d+) days)\b", cleaned, re.I)
    if m_recent:
        days = int(m_recent.group(1) or m_recent.group(2))
        filters["recent_days"] = days
        cleaned = cleaned[:m_recent.start()] + " " + cleaned[m_recent.end():]

    # Pattern: only:cards or in knowledge cards
    m_cards = re.search(r"\b(?:only:cards|only\s+knowledge\s+cards|in\s+knowledge\s+cards)\b", cleaned, re.I)
    if m_cards:
        filters["only_cards"] = True
        cleaned = cleaned[:m_cards.start()] + " " + cleaned[m_cards.end():]

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else query.strip(), filters


def select_diverse_chunks(
    scored_candidates: list[tuple[float, dict[str, Any]]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Select chunks enforcing source and section diversity.

    Prioritizes:
    - Pass 1: Top result from Source A, Top result from Source B, ...
    - Pass 2: Top result from Source A (different section/anchor), ...
    - Pass 3: Remaining highest-scoring chunks that aren't near-duplicates.
    """
    if not scored_candidates:
        return []

    selected: list[dict[str, Any]] = []
    selected_cids: set[str] = set()
    source_counts: dict[str, int] = {}
    source_anchors: dict[str, set[str]] = {}

    # Pass 1: Top result from each distinct source
    for score, chunk in scored_candidates:
        if len(selected) >= limit:
            break
        src_id = chunk.get("source_id", "unknown")
        cid = chunk.get("chunk_id") or chunk.get("id")
        anchor = chunk.get("anchor") or ""

        if src_id not in source_counts:
            selected.append(chunk)
            selected_cids.add(cid)
            source_counts[src_id] = 1
            source_anchors.setdefault(src_id, set()).add(anchor)

    # Pass 2: Next best results from sources, requiring different sections/anchors
    for score, chunk in scored_candidates:
        if len(selected) >= limit:
            break
        cid = chunk.get("chunk_id") or chunk.get("id")
        if cid in selected_cids:
            continue
        src_id = chunk.get("source_id", "unknown")
        anchor = chunk.get("anchor") or ""

        if source_counts.get(src_id, 0) < 2 and anchor not in source_anchors.get(src_id, set()):
            selected.append(chunk)
            selected_cids.add(cid)
            source_counts[src_id] = source_counts.get(src_id, 0) + 1
            source_anchors.setdefault(src_id, set()).add(anchor)

    # Pass 3: Fill remaining quota with highest remaining scoring chunks
    for score, chunk in scored_candidates:
        if len(selected) >= limit:
            break
        cid = chunk.get("chunk_id") or chunk.get("id")
        if cid not in selected_cids:
            selected.append(chunk)
            selected_cids.add(cid)

    return selected


def build_grounded_context_and_citations(
    chunks: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Format the grounded chat context and structured citations for up to 6 chunks.

    Block format:
    [S1] filename — page/section/anchor
    chunk text
    """
    if not chunks:
        return "I found some loosely related material, but not enough evidence for a reliable answer.", []

    final_chunks = chunks[:6]
    context_blocks: list[str] = []
    citations: list[dict[str, Any]] = []

    for idx, c in enumerate(final_chunks, start=1):
        filename = c.get("filename") or "Untitled"
        anchor = c.get("anchor") or f"Chunk {idx}"
        text = c.get("text", "").strip()
        excerpt = text[:180].strip() + ("…" if len(text) > 180 else "")

        context_blocks.append(f"[S{idx}] {filename} — {anchor}\n{text}")
        citations.append({
            "source_id": c.get("source_id", ""),
            "filename": filename,
            "anchor": anchor,
            "chunk_id": c.get("chunk_id") or c.get("id", ""),
            "excerpt": excerpt,
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

    # Local CPU embedding mode: when model is 'local-cpu-simhash', 'local-eval', or no API key is configured,
    # generate deterministic 256-dim signed random projection vectors on CPU and store in SQLite.
    is_local_cpu_embed = (
        model in {"local-eval", "local-cpu-simhash"}
        or not api_key
    )
    if is_local_cpu_embed:
        local_model_name = model if model in {"local-cpu-simhash", "local-eval"} else "local-cpu-simhash"
        newly_local = 0
        for c in to_embed:
            h = c["content_hash"]
            vec = generate_local_embedding(c["text"])
            dims = len(vec)
            vec_json = json.dumps(vec)
            known_vectors[h] = (dims, vec_json)
            db.execute(
                """INSERT INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(chunk_id) DO UPDATE SET
                     model=excluded.model, dimensions=excluded.dimensions,
                     vector_json=excluded.vector_json, content_hash=excluded.content_hash""",
                (str(uuid.uuid4()), c["chunk_id"], local_model_name, dims, vec_json, h, now()),
            )
            newly_local += 1
        msg = "Embedded locally on CPU" if api_key else "Embeddings skipped external API (computed locally on CPU): no API key configured"
        return newly_local, reused_count, msg

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


def _build_metadata_filter_sql(
    filters: dict[str, Any] | None,
    table_alias: str = "c",
    source_alias: str = "s",
) -> tuple[list[str], list[Any]]:
    """Build SQL WHERE clauses and parameter bindings for metadata filters."""
    clauses: list[str] = []
    params: list[Any] = []
    if not filters:
        return clauses, params

    if filters.get("source_id"):
        clauses.append(f"{table_alias}.source_id = ?")
        params.append(filters["source_id"])

    if filters.get("source_filename"):
        clauses.append(f"({source_alias}.filename = ? OR {source_alias}.filename LIKE ?)")
        params.extend([filters["source_filename"], f"%{filters['source_filename']}%"])

    if filters.get("recent_days"):
        since = (datetime.now(UTC) - timedelta(days=int(filters["recent_days"]))).isoformat()
        clauses.append(f"{source_alias}.created_at >= ?")
        params.append(since)

    if filters.get("topic_id"):
        clauses.append(
            f"EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_topics') "
            f"AND {table_alias}.source_id IN (SELECT source_id FROM source_topics WHERE topic_id = ?)"
        )
        params.append(filters["topic_id"])

    if filters.get("topic_name"):
        clauses.append(
            f"({table_alias}.source_id IN ("
            f"SELECT st.source_id FROM source_topics st JOIN topics t ON t.id = st.topic_id "
            f"WHERE t.name = ? COLLATE NOCASE OR t.name LIKE ? COLLATE NOCASE"
            f") OR {source_alias}.filename LIKE ?)"
        )
        t_name = filters["topic_name"].strip()
        params.extend([t_name, f"%{t_name}%", f"%{t_name}%"])

    return clauses, params


def query_knowledge_cards(
    db: Any,
    query: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve matching knowledge cards formatted as chunk candidates when `only_cards` filter is active."""
    try:
        has_cards = db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='cards'").fetchone()
        if not has_cards:
            return []
    except Exception:
        return []

    clauses = ["c.status = 'active'"]
    params: list[Any] = []
    filter_clauses, filter_params = _build_metadata_filter_sql(filters, table_alias="c", source_alias="s")
    clauses.extend(filter_clauses)
    params.extend(filter_params)

    where_sql = " AND ".join(clauses)
    rows = db.execute(
        f"""SELECT c.id AS chunk_id, c.source_id, 1 AS ordinal,
                   (c.title || ': ' || c.body) AS text,
                   ('Card: ' || c.title) AS anchor,
                   '' AS content_hash, s.filename, 0.0 AS bm25_score
            FROM cards c
            JOIN sources s ON s.id = c.source_id
            WHERE {where_sql}""",
        params,
    ).fetchall()

    q_words = [w for w in re.findall(r"[\w-]{2,}", query.lower()) if w not in STOP_WORDS]
    scored: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        d = dict(r)
        txt_lower = d["text"].lower()
        hits = sum(1 for w in q_words if w in txt_lower) if q_words else 1
        if hits > 0 or not q_words:
            scored.append((hits / max(len(q_words), 1), d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:limit]]


def query_fts_chunks(
    db: Any,
    query: str,
    limit: int = 4,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top chunks from SQLite FTS5 with optional metadata filtering."""
    if filters and filters.get("only_cards"):
        return query_knowledge_cards(db, query, limit=limit, filters=filters)

    raw_terms = re.findall(r"[\w-]{2,}", query)
    if not raw_terms:
        return []
    filtered = [term for term in raw_terms if term.lower() not in STOP_WORDS]
    if not filtered:
        return []
    terms = filtered
    match = " OR ".join(f'"{term}"' for term in terms[:12])

    where_clauses = ["chunks_fts MATCH ?"]
    params: list[Any] = [match]

    meta_clauses, meta_params = _build_metadata_filter_sql(filters, table_alias="c", source_alias="s")
    where_clauses.extend(meta_clauses)
    params.extend(meta_params)

    where_sql = " AND ".join(where_clauses)
    params.append(limit)

    try:
        rows = db.execute(
            f"""SELECT c.id AS chunk_id, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename,
                      bm25(chunks_fts) AS bm25_score
               FROM chunks_fts f
               JOIN chunks c ON c.id=f.chunk_id
               JOIN sources s ON s.id=c.source_id
               WHERE {where_sql}
               ORDER BY bm25(chunks_fts) LIMIT ?""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]
    except Exception:
        fallback_params: list[Any] = [f"%{query}%"]
        fallback_clauses = ["c.text LIKE ?"]
        fallback_clauses.extend(meta_clauses)
        fallback_params.extend(meta_params)
        fallback_params.append(limit)
        try:
            rows = db.execute(
                f"""SELECT c.id AS chunk_id, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename, 0.0 AS bm25_score
                   FROM chunks c
                   JOIN sources s ON s.id=c.source_id
                   WHERE {" AND ".join(fallback_clauses)} LIMIT ?""",
                fallback_params,
            ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []


def compute_term_proximity_and_coverage(query: str, text: str) -> float:
    """Calculate query term coverage and sentence-level proximity in passage text."""
    q_words = [w for w in re.findall(r"[\w-]{2,}", query.lower()) if w not in STOP_WORDS]
    if not q_words:
        return 0.0

    t_lower = text.lower()
    t_tokens = set(re.findall(r"[\w-]{2,}", t_lower))

    def _word_matches(w: str, token_set: set[str], raw_str: str) -> bool:
        if w in token_set:
            return True
        if "-" in w and w in raw_str:
            return True
        if len(w) >= 5:
            stem = w[:5]
            return any(tok.startswith(stem) for tok in token_set if len(tok) >= 5)
        return False

    matched_words = [w for w in q_words if _word_matches(w, t_tokens, t_lower)]
    coverage = len(matched_words) / len(q_words)
    if coverage == 0.0:
        return 0.0

    sentences = re.split(r"(?<=[.!?])\s+|\n+", t_lower)
    max_sentence_coverage = 0.0
    for s in sentences:
        s_tokens = set(re.findall(r"[\w-]{2,}", s))
        s_matched = sum(1 for w in q_words if _word_matches(w, s_tokens, s))
        s_cov = s_matched / len(q_words)
        if s_cov > max_sentence_coverage:
            max_sentence_coverage = s_cov

    phrase_bonus = 0.20 if query.lower().strip() in t_lower else 0.0
    return min(1.0, 0.45 * coverage + 0.45 * max_sentence_coverage + phrase_bonus)


def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = 6,
) -> list[tuple[float, dict[str, Any]]]:
    """Lightweight 2nd-stage reranker.

    Base hybrid score follows the tuned weighting:
      final_score = 0.55 * semantic_similarity + 0.35 * keyword_relevance + 0.10 * source_title_match

    Where:
    - `semantic_similarity` combines dense vector cosine similarity with local concept similarity
    - `keyword_relevance` combines FTS5 BM25 rank/coverage and sentence-level term proximity
    - `source_title_match` measures query alignment with source filename and section anchor
    - Exact concepts (e.g. 'TF-IDF', 'BM25', 'CAP') shift weight toward keyword relevance (0.35 sem + 0.55 kw + 0.10 title).
    """
    scored: list[tuple[float, dict[str, Any]]] = []
    q_words = [w for w in re.findall(r"[\w-]{2,}", query.lower()) if w not in STOP_WORDS]
    is_exact_concept = has_exact_concept_token(query)
    q_local_vec = generate_local_embedding(query)

    for item in candidates:
        chunk = item.get("chunk", item)
        raw_sem_score = float(item.get("sem_score", 0.0))
        raw_fts_score = float(item.get("fts_score", 0.0))
        text = chunk.get("text", "")

        # Compute local semantic similarity if external embeddings were not provided
        if raw_sem_score <= 0.0 and text:
            c_local_vec = generate_local_embedding(text)
            sem_score = cosine_similarity(q_local_vec, c_local_vec)
        else:
            sem_score = raw_sem_score

        prox_score = compute_term_proximity_and_coverage(query, text)
        exact_boost = exact_keyword_match_boost(query, text)

        # Effective keyword relevance combines FTS match scaled by actual term coverage + proximity
        if raw_fts_score > 0.0 and prox_score > 0.0:
            keyword_relevance = min(1.0, 0.50 * (raw_fts_score * min(1.0, prox_score + 0.35)) + 0.50 * prox_score + exact_boost)
        else:
            keyword_relevance = min(1.0, 0.80 * prox_score + exact_boost)

        filename = (chunk.get("filename") or "").lower()
        anchor = (chunk.get("anchor") or "").lower()
        matches_title = sum(1 for w in q_words if w in filename or w in anchor)
        title_score = min(1.0, (matches_title / len(q_words)) * 1.5) if q_words else 0.0

        # Apply tuned hybrid formula:
        # Meaning-based questions favor semantic (0.55 sem + 0.35 kw + 0.10 title)
        # Exact concepts like "TF-IDF" favor keyword (0.35 sem + 0.55 kw + 0.10 title)
        if is_exact_concept and keyword_relevance >= 0.35:
            score = (
                0.35 * sem_score
                + 0.55 * keyword_relevance
                + 0.10 * title_score
            )
        else:
            score = (
                0.55 * sem_score
                + 0.35 * keyword_relevance
                + 0.10 * title_score
            )

        # Dual-signal confirmation bonus when both semantic and lexical evidence strongly agree
        if sem_score >= 0.35 and keyword_relevance >= 0.35:
            score = min(1.0, score + 0.08)

        chunk["rerank_score"] = round(score, 4)
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def retrieve_hybrid_chunks(
    db: Any,
    query: str,
    base_url: str,
    api_key: str | None,
    embedding_model: str,
    limit: int = 6,
    filters: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Tuned 2-stage hybrid retrieval with lightweight reranker:
    1. Extracts metadata filters (source file, topic, recency, only_cards) if present.
    2. Stage 1 (Broad Retrieval): Retrieves 10-15 candidates combining FTS5 + vector similarity.
    3. Stage 2 (Reranker): Applies multi-signal similarity check (0.55 sem + 0.35 kw + 0.10 title).
    4. Stage 3 (Gating & Diversity): Applies CONFIDENCE_THRESHOLD (0.30) and selects 4-6 diverse chunks.
    """
    clean_query, inline_filters = extract_query_filters(query)
    active_filters = dict(inline_filters)
    if filters:
        for k, v in filters.items():
            if v is not None and v != "":
                active_filters[k] = v

    if active_filters.get("only_cards"):
        card_candidates = query_knowledge_cards(db, clean_query, limit=12, filters=active_filters)
        raw_card_items = [
            {"chunk": c, "fts_score": max(0.25, 0.85 - idx * 0.08), "sem_score": 0.0}
            for idx, c in enumerate(card_candidates)
        ]
        reranked_cards = rerank_candidates(clean_query, raw_card_items)
        if not reranked_cards or reranked_cards[0][0] < CONFIDENCE_THRESHOLD:
            return []
        return select_diverse_chunks(reranked_cards, limit=limit)

    stage1_limit = 10
    fts_candidates = query_fts_chunks(db, clean_query, limit=stage1_limit, filters=active_filters)

    meta_clauses, meta_params = _build_metadata_filter_sql(active_filters, table_alias="c", source_alias="s")

    # Fetch stored vectors if available (configured model first, then local-cpu-simhash / local-eval fallback)
    stored_vectors: list[Any] = []
    if embedding_model:
        where_vec = ["e.model=?"] + meta_clauses
        params_vec: list[Any] = [embedding_model] + meta_params
        try:
            stored_vectors = db.execute(
                f"""SELECT e.chunk_id, e.model, e.vector_json, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename
                   FROM embeddings e
                   JOIN chunks c ON c.id=e.chunk_id
                   JOIN sources s ON s.id=c.source_id
                   WHERE {" AND ".join(where_vec)}""",
                params_vec,
            ).fetchall()
            if not stored_vectors and embedding_model not in {"local-cpu-simhash", "local-eval"}:
                where_fallback = ["e.model IN ('local-cpu-simhash', 'local-eval')"] + meta_clauses
                stored_vectors = db.execute(
                    f"""SELECT e.chunk_id, e.model, e.vector_json, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename
                       FROM embeddings e
                       JOIN chunks c ON c.id=e.chunk_id
                       JOIN sources s ON s.id=c.source_id
                       WHERE {" AND ".join(where_fallback)}""",
                    meta_params,
                ).fetchall()
        except Exception:
            stored_vectors = []

    query_vec: list[float] | None = None
    first_model = stored_vectors[0]["model"] if stored_vectors and "model" in stored_vectors[0].keys() else embedding_model
    if (api_key or "localhost" in base_url or "127.0.0.1" in base_url) and stored_vectors and first_model not in {"local-eval", "local-cpu-simhash"}:
        try:
            query_vectors = call_openai_embeddings_api(base_url, api_key or "local-cpu", embedding_model, [clean_query])
            query_vec = query_vectors[0] if query_vectors else None
        except Exception:
            query_vec = None
    elif stored_vectors and first_model in {"local-eval", "local-cpu-simhash"}:
        query_vec = generate_local_embedding(clean_query)

    # If no external vectors exist in DB, compute local semantic candidates across filtered chunks
    semantic_top: list[tuple[float, dict[str, Any]]] = []
    if query_vec and stored_vectors:
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
        semantic_top = semantic_scored[:stage1_limit]
    else:
        # Local semantic search over SQLite chunks when no API key is set
        local_qvec = generate_local_embedding(clean_query)
        where_all = ["1=1"] + meta_clauses
        try:
            all_chunks = db.execute(
                f"""SELECT c.id AS chunk_id, c.source_id, c.ordinal, c.text, c.anchor, c.content_hash, s.filename
                   FROM chunks c
                   JOIN sources s ON s.id=c.source_id
                   WHERE {" AND ".join(where_all)}""",
                meta_params,
            ).fetchall()
            local_scored: list[tuple[float, dict[str, Any]]] = []
            for row in all_chunks:
                item = dict(row)
                sim = cosine_similarity(local_qvec, generate_local_embedding(item.get("text", "")))
                if sim > 0.05:
                    local_scored.append((sim, item))
            local_scored.sort(key=lambda x: x[0], reverse=True)
            semantic_top = local_scored[:stage1_limit]
        except Exception:
            semantic_top = []

    # Combine FTS top 10 + semantic top 10 into Stage 1 candidate pool (10-15 chunks)
    candidates: dict[str, dict[str, Any]] = {}

    for rank, fts_chunk in enumerate(fts_candidates):
        cid = fts_chunk["chunk_id"]
        candidates[cid] = {
            "chunk": fts_chunk,
            "fts_score": max(0.20, 0.90 - (rank * 0.08)),
            "sem_score": 0.0,
        }

    for sem_score, sem_chunk in semantic_top:
        cid = sem_chunk["chunk_id"]
        if cid in candidates:
            candidates[cid]["sem_score"] = max(0.0, sem_score)
        else:
            candidates[cid] = {
                "chunk": sem_chunk,
                "fts_score": 0.0,
                "sem_score": max(0.0, sem_score),
            }

    candidate_list = list(candidates.values())
    if not candidate_list:
        return []

    # Stage 2: Apply lightweight reranker
    reranked_candidates = rerank_candidates(clean_query, candidate_list)

    # Stage 3: Confidence threshold check
    if not reranked_candidates or reranked_candidates[0][0] < CONFIDENCE_THRESHOLD:
        return []

    # Filter out individual weak tail chunks below threshold
    qualified = [(score, c) for score, c in reranked_candidates if score >= CONFIDENCE_THRESHOLD]
    if not qualified:
        return []

    # Select final 4-6 diverse chunks
    return select_diverse_chunks(qualified, limit=limit)
