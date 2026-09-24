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

    def test_text_capture_is_searchable_and_keeps_a_managed_copy(self):
        response = self.client.post("/api/captures/text", json={"title": "RAG", "content": "Retrieval augmented generation uses relevant documents as context."})
        self.assertEqual(response.status_code, 200)
        source_id = response.json()["id"]
        sources = self.client.get("/api/sources").json()
        self.assertEqual(sources[0]["id"], source_id)
        self.assertEqual(sources[0]["status"], "ready")
        self.assertTrue(Path(sources[0]["managed_path"]).exists())
        search = self.client.get("/api/search?q=retrieval").json()
        self.assertTrue(search)

    def test_duplicate_capture_is_reported(self):
        payload = {"title": "Same", "content": "A duplicate body."}
        self.client.post("/api/captures/text", json=payload)
        response = self.client.post("/api/captures/text", json=payload)
        self.assertEqual(response.json()["status"], "duplicate")

