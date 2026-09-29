import os
import tempfile
import unittest
from pathlib import Path

TEST_DIR = tempfile.mkdtemp(prefix="ren-test-")
os.environ["REN_DATA_DIR"] = TEST_DIR

from fastapi.testclient import TestClient
from backend.app import app


class RenApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_text_capture_creates_due_knowledge_cards_and_keeps_a_managed_copy(self):
        response = self.client.post("/api/captures/text", json={"title": "RAG", "content": "Retrieval augmented generation uses relevant documents as context."})
        self.assertEqual(response.status_code, 200)
        source_id = response.json()["id"]
        sources = self.client.get("/api/sources").json()
        self.assertEqual(sources[0]["id"], source_id)
        self.assertEqual(sources[0]["status"], "ready")
        self.assertTrue(Path(sources[0]["managed_path"]).exists())
        cards = self.client.get("/api/cards").json()
        self.assertTrue(cards)
        self.assertEqual(cards[0]["source"]["id"], source_id)
        queue = self.client.get("/api/remember").json()
        self.assertTrue(queue)

    def test_review_moves_card_out_of_remember_queue(self):
        self.client.post("/api/captures/text", json={"title": "Dijkstra", "content": "Dijkstra's algorithm finds shortest paths in graphs with non-negative edge weights."})
        card = self.client.get("/api/remember").json()[0]
        response = self.client.post(f"/api/cards/{card['id']}/review")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["review_count"], 1)
        queue_ids = {item["id"] for item in self.client.get("/api/remember").json()}
        self.assertNotIn(card["id"], queue_ids)

    def test_markdown_headings_become_named_knowledge_cards(self):
        content = "# Retrieval\n\nRetrieval finds useful documents before an answer is generated.\n\n## Embeddings\n\nEmbeddings map text into vectors so similar meanings can be compared."
        response = self.client.post("/api/captures/text", json={"title": "rag headings", "content": content})
        self.assertEqual(response.status_code, 200)
        cards = self.client.get("/api/cards").json()
        titles = {card["title"] for card in cards}
        self.assertIn("Retrieval", titles)
        self.assertIn("Embeddings", titles)

    def test_duplicate_capture_is_reported(self):
        payload = {"title": "Same", "content": "A duplicate body."}
        self.client.post("/api/captures/text", json=payload)
        response = self.client.post("/api/captures/text", json=payload)
        self.assertEqual(response.json()["status"], "duplicate")

