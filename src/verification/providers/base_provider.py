"""Abstract LLM Provider Base Class for CiteGuard."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import json
import re
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract interface for LLM provider implementations (Mock, OpenAI, Gemini, Ollama)."""

    def __init__(
        self,
        model_name: str = "mock-model",
        temperature: float = 0.0,
        timeout_sec: float = 30.0,
        max_retries: int = 3,
        **kwargs
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.call_count = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text response given a prompt and optional system prompt."""
        pass

    def generate_structured(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate response and attempt to parse JSON structure from output."""
        raw_response = self.generate(prompt, system_prompt=system_prompt)
        return self._extract_json(raw_response)

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Utility to extract JSON object from markdown or raw text output."""
        text = text.strip()
        # Try direct json parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extract from ```json ... ``` blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding raw curly braces
        curly_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if curly_match:
            try:
                return json.loads(curly_match.group(1))
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse JSON from provider output: {text[:100]}...")
        return {"raw_output": text, "error": "json_parse_failed"}

    def reset_stats(self):
        """Reset execution statistics counters."""
        self.call_count = 0
        self.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
