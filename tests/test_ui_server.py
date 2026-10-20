"""Test Suite for CiteGuard FastAPI Web Backend Server."""

import os
import unittest
from fastapi.testclient import TestClient

from src.ui.server import app, REPORTS_DIR


class TestUiServer(unittest.TestCase):
    """Integration test suite for CiteGuard Web UI Server."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_1_get_frontend_root(self):
        """Test serving root URL index.html."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("CiteGuard", response.text)
        self.assertIn("OpenAlex + Local Corpus", response.text)

    def test_2_get_system_status(self):
        """Test system status endpoint."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["provider_display"], "OpenAlex + Local Corpus")

    def test_3_verify_endpoint_success(self):
        """Test POST /api/verify with valid scientific question."""
        payload = {
            "question": "What is the relationship between sleep duration and cognitive performance?",
            "provider": "mock",
            "top_k": 5,
            "mode": "hybrid"
        }
        response = self.client.post("/api/verify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("question_id", data)
        self.assertIn("report", data)
        self.assertEqual(len(data["report"]["claims"]), 3)

    def test_4_verify_endpoint_invalid_question(self):
        """Test POST /api/verify with invalid short question."""
        payload = {
            "question": "Short?",
            "provider": "mock"
        }
        response = self.client.post("/api/verify", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)

    def test_5_download_report_file(self):
        """Test GET /api/reports/{filename} endpoint."""
        # Ensure at least one report file exists in REPORTS_DIR
        files = [f for f in os.listdir(REPORTS_DIR) if f.endswith(".json")]
        if files:
            filename = files[0]
            response = self.client.get(f"/api/reports/{filename}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "application/json")


if __name__ == "__main__":
    unittest.main()
