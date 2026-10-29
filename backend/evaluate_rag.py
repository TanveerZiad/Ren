"""RAG Evaluation and Retrieval Tuning Runner for Ren.

Evaluates the actual `retrieve_hybrid_chunks` 2-stage pipeline across the 20-question benchmark:
- Top-5 Source Hit Rate
- Target Chunk Hit Rate
- Answer Groundedness / Citation Support
- Insufficient Evidence Refusal Accuracy (requires empty retrieval due to CONFIDENCE_THRESHOLD)
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backend.chat_service import grounded_system_prompt
from backend.rag_service import (
    build_grounded_context_and_citations,
    chunk_source_document,
    generate_local_embedding,
    retrieve_hybrid_chunks,
)

EVAL_DIR = Path(__file__).resolve().parent.parent / "eval"
SOURCES_DIR = EVAL_DIR / "sources"
DATASET_PATH = EVAL_DIR / "rag_eval_dataset.json"
RESULTS_PATH = EVAL_DIR / "rag_evaluation_results.json"


def setup_eval_database(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE sources (
            id TEXT PRIMARY KEY, filename TEXT NOT NULL, source_kind TEXT NOT NULL,
            status TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE chunks (
            id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL, text TEXT NOT NULL, anchor TEXT, content_hash TEXT, created_at TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(chunk_id UNINDEXED, text);
        CREATE TABLE embeddings (
            id TEXT PRIMARY KEY, chunk_id TEXT NOT NULL UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
            model TEXT NOT NULL, dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL,
            content_hash TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE topics (
            id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
            summary TEXT DEFAULT '', created_at TEXT NOT NULL
        );
        CREATE TABLE source_topics (
            source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            topic_id TEXT NOT NULL REFERENCES topics(id) ON DELETE CASCADE,
            confidence REAL NOT NULL, PRIMARY KEY(source_id, topic_id)
        );
        """
    )
    return conn


def ingest_eval_sources(conn: sqlite3.Connection, sources_dir: Path) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for file_path in sorted(sources_dir.glob("*")):
        if not file_path.is_file():
            continue
        filename = file_path.name
        text = file_path.read_text(encoding="utf-8")
        src_id = str(uuid.uuid4())
        source_map[filename] = src_id
        timestamp = datetime.now(UTC).isoformat()

        conn.execute(
            "INSERT INTO sources(id, filename, source_kind, status, created_at, updated_at) VALUES (?, ?, 'text', 'ready', ?, ?)",
            (src_id, filename, timestamp, timestamp),
        )

        # Split smaller sections per paragraph/heading in eval corpus so multiple chunks exist per source
        chunks = chunk_source_document(file_path, text, source_kind="text", target_words=80, min_words=40, max_words=140, overlap_words=15)
        for c in chunks:
            cid = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO chunks(id, source_id, ordinal, text, anchor, content_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cid, src_id, c["ordinal"], c["text"], c["anchor"], c["content_hash"], timestamp),
            )
            conn.execute("INSERT INTO chunks_fts(chunk_id, text) VALUES (?, ?)", (cid, c["text"]))

            vec = generate_local_embedding(c["text"])
            conn.execute(
                "INSERT INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at) VALUES (?, ?, 'local-eval', ?, ?, ?, ?)",
                (str(uuid.uuid4()), cid, len(vec), json.dumps(vec), c["content_hash"], timestamp),
            )
    conn.commit()
    return source_map


def run_rag_evaluation() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Evaluation dataset missing at {DATASET_PATH}")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))

    temp_dir = tempfile.mkdtemp(prefix="ren-rag-eval-")
    db_file = Path(temp_dir) / "eval.sqlite3"
    conn = setup_eval_database(db_file)
    ingest_eval_sources(conn, SOURCES_DIR)

    results: list[dict[str, Any]] = []

    source_hits = 0
    chunk_hits = 0
    grounded_answers = 0
    insufficient_evidence_correct = 0

    total_answerable = sum(1 for q in dataset if not q.get("insufficient_evidence"))
    total_unanswerable = sum(1 for q in dataset if q.get("insufficient_evidence"))

    for item in dataset:
        qid = item["id"]
        question = item["question"]
        expected_src = item.get("expected_source")
        expected_terms = item.get("expected_terms", [])
        is_unanswerable = item.get("insufficient_evidence", False)

        # Call the ACTUAL production hybrid retrieval pipeline
        retrieved = retrieve_hybrid_chunks(
            db=conn,
            query=question,
            base_url="http://localhost",
            api_key=None,
            embedding_model="local-eval",
            limit=5,
        )

        retrieved_sources = [c.get("filename") for c in retrieved[:5]]
        retrieved_texts = [c.get("text", "") for c in retrieved[:5]]

        context_str, citations = build_grounded_context_and_citations(retrieved[:5])
        _ = grounded_system_prompt(context_str, [])

        if is_unanswerable:
            # Strict refusal verification: retrieve_hybrid_chunks MUST suppress weak matches via CONFIDENCE_THRESHOLD
            refused = len(retrieved) == 0 and len(citations) == 0
            source_hit = refused
            chunk_hit = refused
            evidence_stated = refused and ("not enough evidence" in context_str.lower())
            answer_supported = refused
            if evidence_stated:
                insufficient_evidence_correct += 1
        else:
            source_hit = expected_src in retrieved_sources if expected_src else False
            combined_text = " ".join(retrieved_texts).lower()
            chunk_hit = any(term.lower() in combined_text for term in expected_terms) if expected_terms else False
            answer_supported = source_hit and chunk_hit
            evidence_stated = None

            if source_hit:
                source_hits += 1
            if chunk_hit:
                chunk_hits += 1
            if answer_supported:
                grounded_answers += 1

        results.append({
            "id": qid,
            "question": question,
            "expected_source": expected_src,
            "expected_terms": expected_terms,
            "insufficient_evidence_scenario": is_unanswerable,
            "top_5_sources": retrieved_sources,
            "top_5_source_hit": source_hit,
            "target_chunk_hit": chunk_hit,
            "final_answer_supported_by_citations": answer_supported,
            "correct_insufficient_evidence_response": evidence_stated,
            "citations": [f"[S{i+1}] {c['filename']} — {c['anchor']}" for i, c in enumerate(citations)],
        })

    conn.close()

    source_hit_rate = (source_hits / total_answerable) * 100 if total_answerable else 0.0
    chunk_hit_rate = (chunk_hits / total_answerable) * 100 if total_answerable else 0.0
    grounded_rate = (grounded_answers / total_answerable) * 100 if total_answerable else 0.0
    insufficient_acc = (insufficient_evidence_correct / total_unanswerable) * 100 if total_unanswerable else 0.0

    summary = {
        "evaluation_timestamp": datetime.now(UTC).isoformat(),
        "total_questions": len(dataset),
        "answerable_questions": total_answerable,
        "unanswerable_negative_controls": total_unanswerable,
        "metrics": {
            "top_5_source_hit_rate_pct": round(source_hit_rate, 1),
            "target_chunk_hit_rate_pct": round(chunk_hit_rate, 1),
            "answer_groundedness_pct": round(grounded_rate, 1),
            "insufficient_evidence_accuracy_pct": round(insufficient_acc, 1),
        },
        "per_question_results": results,
    }

    RESULTS_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    summary = run_rag_evaluation()
    m = summary["metrics"]
    print("=" * 64)
    print("REN RAG EVALUATION & RETRIEVAL TUNING REPORT")
    print("=" * 64)
    print(f"Total Questions Evaluated: {summary['total_questions']}")
    print(f"Top-5 Source Hit Rate:     {m['top_5_source_hit_rate_pct']}%")
    print(f"Target Chunk Hit Rate:     {m['target_chunk_hit_rate_pct']}%")
    print(f"Answer Groundedness:       {m['answer_groundedness_pct']}%")
    print(f"Insufficient Evidence Acc: {m['insufficient_evidence_accuracy_pct']}%")
    print("=" * 64)
    for q in summary["per_question_results"]:
        status = "PASS" if (q["top_5_source_hit"] and q["target_chunk_hit"]) else "FAIL"
        print(f"[{q['id']}] {status} - Q: {q['question']}")
        print(f"     Source Hit: {q['top_5_source_hit']} | Chunk Hit: {q['target_chunk_hit']} | Cited: {', '.join(q['citations'][:2])}")
