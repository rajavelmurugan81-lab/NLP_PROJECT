"""Unit tests for OllamaProvider."""

import json
import unittest
from unittest.mock import patch, MagicMock

from src.verification.providers.ollama_provider import OllamaProvider
from src.verification.providers.factory import get_provider


class TestOllamaProvider(unittest.TestCase):
    """Test suite for OllamaProvider LLM integration."""

    def setUp(self):
        self.provider = OllamaProvider(model_name="llama3.2", base_url="http://localhost:11434/v1/chat/completions")

    @patch("urllib.request.urlopen")
    def test_generate_openai_format(self, mock_urlopen):
        # Mock response format for /v1/chat/completions
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "{\"verdict\": \"SUPPORTED\", \"confidence\": 0.9}"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8
            }
        }
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        res = self.provider.generate("Test prompt")
        self.assertIn("SUPPORTED", res)
        self.assertEqual(self.provider.call_count, 1)
        self.assertEqual(self.provider.token_usage["total_tokens"], 23)

    @patch("urllib.request.urlopen")
    def test_generate_native_ollama_format(self, mock_urlopen):
        native_provider = OllamaProvider(model_name="llama3.2", base_url="http://localhost:11434/api/chat")
        mock_response_data = {
            "message": {
                "role": "assistant",
                "content": "{\"verdict\": \"REFUTED\", \"confidence\": 0.85}"
            },
            "prompt_eval_count": 20,
            "eval_count": 10
        }
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        res = native_provider.generate("Test prompt")
        self.assertIn("REFUTED", res)
        self.assertEqual(native_provider.call_count, 1)

    @patch("urllib.request.urlopen")
    def test_generate_structured_json(self, mock_urlopen):
        mock_response_data = {
            "choices": [
                {
                    "message": {
                        "content": "{\"verdict\": \"SUPPORTED\", \"reasoning\": \"Valid evidence\"}"
                    }
                }
            ]
        }
        mock_cm = MagicMock()
        mock_cm.read.return_value = json.dumps(mock_response_data).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = mock_cm

        res_dict = self.provider.generate_structured("Test prompt")
        self.assertEqual(res_dict.get("verdict"), "SUPPORTED")
        self.assertEqual(res_dict.get("reasoning"), "Valid evidence")

    def test_factory_instantiation(self):
        prov = get_provider("ollama", model_name="mistral")
        self.assertIsInstance(prov, OllamaProvider)
        self.assertEqual(prov.model_name, "mistral")


if __name__ == "__main__":
    unittest.main()
