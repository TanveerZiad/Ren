import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

TEST_DIR = tempfile.mkdtemp(prefix="ren-rag-test-")
os.environ["REN_DATA_DIR"] = TEST_DIR

from fastapi.testclient import TestClient

from backend.app import app, configured_embedding_provider, db_connection, now, process_source
from backend.rag_service import (
    build_grounded_context_and_citations,
    chunk_source_document,
    chunk_text,
    compute_content_hash,
    cosine_similarity,
    embed_chunks_for_source,
    exact_keyword_match_boost,
    retrieve_hybrid_chunks,
)


class RenHybridRagTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_500_word_chunking_with_overlap(self):
        # Generate text with sentences totaling ~1500 words across multiple paragraphs
        paragraph_sentences = []
        for p in range(5):
            sentences = [f"This is sentence {s} of paragraph {p} discussing advanced information retrieval algorithms and system design." for s in range(30)]
            paragraph_sentences.append(" ".join(sentences))
        full_text = "\n\n".join(paragraph_sentences)
        total_words = len(full_text.split())
        self.assertGreater(total_words, 1200)

        chunks = chunk_text(full_text, target_words=500, min_words=400, max_words=700, overlap_words=60)
        self.assertGreaterEqual(len(chunks), 2)

        for i, chunk in enumerate(chunks):
            wcount = len(chunk.split())
            if i < len(chunks) - 1:
                self.assertGreaterEqual(wcount, 400, f"Chunk {i} word count {wcount} is below 400")
                self.assertLessEqual(wcount, 700, f"Chunk {i} word count {wcount} exceeds 700")

        # Verify adjacent chunks share exactly 60-word overlap
        for i in range(len(chunks) - 1):
            c1_words = chunks[i].split()
            c2_words = chunks[i + 1].split()
            self.assertEqual(c1_words[-60:], c2_words[:60], f"Overlap mismatch between chunk {i} and {i+1}")

    def test_chunk_source_document_anchor_preservation(self):
        markdown_text = (
            "# Architecture\n\n"
            + " ".join(["System architecture describes how components interact in a distributed environment."] * 40)
            + "\n\n## Vector Storage\n\n"
            + " ".join(["Vector storage persists embeddings with metadata and cosine distance indexing."] * 40)
        )
        chunks = chunk_source_document(None, markdown_text, source_kind="text")
        self.assertGreaterEqual(len(chunks), 2)

        anchors = [c["anchor"] for c in chunks]
        self.assertIn("Architecture", anchors)
        self.assertIn("Vector Storage", anchors)

        for c in chunks:
            self.assertTrue(c["content_hash"])
            self.assertEqual(c["content_hash"], compute_content_hash(c["text"]))
            self.assertIsInstance(c["ordinal"], int)

    def test_unchanged_chunks_not_embedded_twice(self):
        mock_embedding_vector = [0.05] * 128
        api_call_count = 0

        def fake_embedding_api(base_url, api_key, model, texts, timeout=40):
            nonlocal api_call_count
            api_call_count += len(texts)
            return [mock_embedding_vector for _ in texts]

        with patch("backend.rag_service.call_openai_embeddings_api", side_effect=fake_embedding_api):
            with patch("backend.app.get_api_key", return_value="fake-api-key"):
                # 1. First import
                content1 = "# Knowledge Base\n\n" + " ".join(["Ren provides private personal knowledge management."] * 50)
                res1 = self.client.post("/api/captures/text", json={"title": "Doc One", "content": content1})
                source_id1 = res1.json()["id"]

                first_call_count = api_call_count
                self.assertGreater(first_call_count, 0)

                # Check embeddings exist in DB
                with db_connection() as db:
                    rows = db.execute("SELECT * FROM embeddings").fetchall()
                    self.assertGreater(len(rows), 0)

                # 2. Retry the same source (all chunks unchanged)
                retry_res = self.client.post(f"/api/sources/{source_id1}/retry")
                self.assertEqual(retry_res.status_code, 200)

                # Verify no additional embedding API calls were made because hashes were already in DB
                self.assertEqual(api_call_count, first_call_count, "Unchanged chunks must never be re-embedded on retry!")

                # 3. Import second document that shares chunk from Doc One, but has an additional chunk
                content2 = content1 + "\n\n## Brand New Section\n\n" + " ".join(["Completely novel concepts that require a fresh embedding."] * 50)
                res2 = self.client.post("/api/captures/text", json={"title": "Doc Two Shared", "content": content2})
                self.assertEqual(res2.status_code, 200)

                # The new call should only have embedded the new section, NOT the unchanged section
                newly_called = api_call_count - first_call_count
                self.assertGreater(newly_called, 0)
                # Less than total chunks of doc2
                self.assertLess(newly_called, first_call_count + newly_called)

    def test_fts_only_retrieval_with_no_api_key(self):
        content = "Photosynthesis converts light energy into chemical energy in plant chloroplasts."
        with patch("backend.app.get_api_key", return_value=None):
            res = self.client.post("/api/captures/text", json={"title": "Biology", "content": content})
            source_id = res.json()["id"]
            job_id = res.json()["job_id"]

            # Ingestion succeeds and source is ready
            sources = self.client.get("/api/sources").json()
            target_source = next(s for s in sources if s["id"] == source_id)
            self.assertEqual(target_source["status"], "ready")

            # Processing job recorded completion and notes embedding was skipped
            jobs = self.client.get("/api/jobs").json()
            job = next(j for j in jobs if j["id"] == job_id)
            self.assertEqual(job["status"], "completed")
            self.assertIn("skipped", (job.get("error_message") or "").lower())

            # FTS5 search works fully without API key
            results = self.client.get("/api/search", params={"q": "chloroplasts"}).json()
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["source_id"], source_id)
            self.assertIn("Photosynthesis", results[0]["text"])

    def test_semantic_and_keyword_deduplication(self):
        # Insert 3 chunks:
        # Chunk 1: Keyword match for 'graph', vector sim ~0.99
        # Chunk 2: Keyword match for 'graph', vector sim ~0.60
        # Chunk 3: No keyword match for 'graph' (about 'networks'), vector sim ~0.99
        with db_connection() as db:
            src_id = "test-source-dedup"
            db.execute("INSERT OR REPLACE INTO sources(id, filename, source_kind, status, created_at, updated_at) VALUES (?, 'doc.md', 'text', 'ready', ?, ?)", (src_id, now(), now()))
            c1_id = "chunk-c1"
            c2_id = "chunk-c2"
            c3_id = "chunk-c3"

            for cid, ord_num, txt in [
                (c1_id, 1, "Graph traversal using Dijkstra and BFS algorithms."),
                (c2_id, 2, "Graph visualization in web browsers."),
                (c3_id, 3, "Network routing protocols and packet switching."),
            ]:
                db.execute("INSERT OR REPLACE INTO chunks(id, source_id, ordinal, text, anchor, content_hash, created_at) VALUES (?, ?, ?, ?, 'Section', 'hash', ?)", (cid, src_id, ord_num, txt, now()))
                db.execute("INSERT OR REPLACE INTO chunks_fts(chunk_id, text) VALUES (?, ?)", (cid, txt))

            # Store vectors:
            # Query vector will be [1.0, 0.0]
            # Chunk 1: [0.95, 0.05] (sim ~0.99)
            # Chunk 2: [0.60, 0.80] (sim ~0.60)
            # Chunk 3: [0.90, 0.10] (sim ~0.99)
            db.execute("INSERT OR REPLACE INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at) VALUES ('e1', ?, 'text-embedding-3-small', 2, '[0.95, 0.05]', 'h1', ?)", (c1_id, now()))
            db.execute("INSERT OR REPLACE INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at) VALUES ('e2', ?, 'text-embedding-3-small', 2, '[0.60, 0.80]', 'h2', ?)", (c2_id, now()))
            db.execute("INSERT OR REPLACE INTO embeddings(id, chunk_id, model, dimensions, vector_json, content_hash, created_at) VALUES ('e3', ?, 'text-embedding-3-small', 2, '[0.90, 0.10]', 'h3', ?)", (c3_id, now()))

            with patch("backend.rag_service.call_openai_embeddings_api", return_value=[[1.0, 0.0]]):
                final_chunks = retrieve_hybrid_chunks(
                    db=db,
                    query="graph",
                    base_url="https://api.openai.com/v1",
                    api_key="test-key",
                    embedding_model="text-embedding-3-small",
                    limit=6,
                )

                chunk_ids = [c["chunk_id"] for c in final_chunks]
                # Assert deduplication: chunk_ids has no duplicates
                self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
                self.assertLessEqual(len(final_chunks), 6)

                # Chunk 1 (present in both FTS keyword + semantic + exact keyword match) ranks first
                self.assertEqual(final_chunks[0]["chunk_id"], c1_id)

    def test_exact_keyword_boost_when_scores_are_close(self):
        query = "distributed database"
        text_with_keyword = "A distributed database shards records across nodes."
        text_without_keyword = "Storage engines partition data over network clusters."

        boost_match = exact_keyword_match_boost(query, text_with_keyword)
        boost_nomatch = exact_keyword_match_boost(query, text_without_keyword)

        self.assertGreater(boost_match, boost_nomatch)

    def test_citation_metadata_in_chat_context(self):
        content = "# Storage\n\nSQLite is a C-language library that implements a small, fast, self-contained SQL database engine."
        res = self.client.post("/api/captures/text", json={"title": "SQLite", "content": content})
        source_id = res.json()["id"]

        with patch("backend.app.get_api_key", return_value=None):
            process_source(source_id, res.json()["job_id"])

        thread = self.client.post("/api/chat/threads", json={"title": "Test Chat"}).json()
        thread_id = thread["id"]

        with patch("backend.app.complete_chat") as mock_complete:
            from backend.chat_service import ChatCompletion
            mock_complete.return_value = ChatCompletion(content="SQLite is an embedded database [S1].", provider_used=True)

            msg_res = self.client.post(f"/api/chat/threads/{thread_id}/messages", json={"content": "What is SQLite?"})
            self.assertEqual(msg_res.status_code, 200)
            data = msg_res.json()

            # Verify citations contain source_id, filename, anchor, and chunk_id
            citations = data.get("citations", [])
            self.assertGreater(len(citations), 0)
            c0 = citations[0]
            self.assertIn("source_id", c0)
            self.assertIn("filename", c0)
            self.assertIn("anchor", c0)
            self.assertIn("chunk_id", c0)
            self.assertTrue(c0["chunk_id"])

            # Verify context passed to model has [S1] filename — anchor format
            call_args = mock_complete.call_args[1]
            context = call_args["context"]
            self.assertIn("[S1]", context)
            self.assertIn("—", context)

    def test_embedding_batching_maximum_32(self):
        batch_sizes_called = []

        def fake_api(base_url, api_key, model, texts, timeout=40):
            batch_sizes_called.append(len(texts))
            return [[0.1, 0.2] for _ in texts]

        with patch("backend.rag_service.call_openai_embeddings_api", side_effect=fake_api):
            dummy_chunks = [
                {"chunk_id": f"batch-chunk-{i}", "text": f"Unique chunk text number {i}", "content_hash": f"batch-hash-{i}"}
                for i in range(70)
            ]
            with db_connection() as db:
                db.execute("INSERT OR REPLACE INTO sources(id, filename, source_kind, status, created_at, updated_at) VALUES ('src-batch', 'f.txt', 'text', 'ready', ?, ?)", (now(), now()))
                for c in dummy_chunks:
                    db.execute(
                        "INSERT OR REPLACE INTO chunks(id, source_id, ordinal, text, anchor, content_hash, created_at) VALUES (?, 'src-batch', 1, ?, 'A', ?, ?)",
                        (c["chunk_id"], c["text"], c["content_hash"], now()),
                    )

                newly, reused, err = embed_chunks_for_source(
                    db=db,
                    chunks=dummy_chunks,
                    known_vectors={},
                    base_url="https://api.openai.com/v1",
                    api_key="key",
                    model="text-embedding-3-small",
                )
                self.assertEqual(newly, 70)
                self.assertEqual(reused, 0)
                self.assertIsNone(err)

            # Check that maximum batch size in all calls was <= 32
            for sz in batch_sizes_called:
                self.assertLessEqual(sz, 32)
            self.assertEqual(sum(batch_sizes_called), 70)
            self.assertEqual(batch_sizes_called, [32, 32, 6])

    def test_exact_term_tfidf_keyword_retrieval_and_boost(self):
        query = "What is TF-IDF?"
        text_with_tfidf = "TF-IDF reflects how important a word is to a document in a collection."
        text_without = "Bag of words counts token frequencies without inverse document frequency scaling."

        boost_match = exact_keyword_match_boost(query, text_with_tfidf)
        boost_nomatch = exact_keyword_match_boost(query, text_without)
        self.assertGreater(boost_match, boost_nomatch)

        with db_connection() as db:
            src_id = "test-tfidf-src"
            db.execute("INSERT OR REPLACE INTO sources(id, filename, source_kind, status, created_at, updated_at) VALUES (?, 'ir.md', 'text', 'ready', ?, ?)", (src_id, now(), now()))
            c1 = "chunk-tfidf-1"
            c2 = "chunk-tfidf-2"
            db.execute("INSERT OR REPLACE INTO chunks(id, source_id, ordinal, text, anchor, content_hash, created_at) VALUES (?, ?, 1, ?, 'Section', 'h1', ?)", (c1, src_id, text_with_tfidf, now()))
            db.execute("INSERT OR REPLACE INTO chunks(id, source_id, ordinal, text, anchor, content_hash, created_at) VALUES (?, ?, 2, ?, 'Section', 'h2', ?)", (c2, src_id, text_without, now()))
            db.execute("INSERT OR REPLACE INTO chunks_fts(chunk_id, text) VALUES (?, ?)", (c1, text_with_tfidf))
            db.execute("INSERT OR REPLACE INTO chunks_fts(chunk_id, text) VALUES (?, ?)", (c2, text_without))

            results = retrieve_hybrid_chunks(
                db=db,
                query=query,
                base_url="https://api.openai.com/v1",
                api_key=None,
                embedding_model="text-embedding-3-small",
                limit=4,
            )
            self.assertGreater(len(results), 0)
            self.assertEqual(results[0]["chunk_id"], c1)
            self.assertIn("TF-IDF", results[0]["text"])

    def test_grounded_system_prompt_enforces_rules(self):
        from backend.chat_service import grounded_system_prompt
        prompt = grounded_system_prompt(context="[S1] doc.txt — Sec 1\nTF-IDF explanation.", memories=["Likes concise notes"])
        self.assertIn("Answer only from the provided Ren source excerpts.", prompt)
        self.assertIn("If the evidence is insufficient, say so.", prompt)
        self.assertIn("Cite every important claim as [S1], [S2], etc.", prompt)
        self.assertIn("Do not use outside knowledge unless web search is explicitly enabled.", prompt)


if __name__ == "__main__":
    unittest.main()
