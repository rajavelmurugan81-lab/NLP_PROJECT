"""Ollama Local LLM Provider implementation for CiteGuard."""

import os
import json
import logging
from typing import Dict, Any, Optional
import urllib.request
import urllib.error

from src.verification.providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama Local LLM Provider supporting HTTP communication with local Ollama service."""

    def __init__(
        self,
        model_name: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1/chat/completions",
        temperature: float = 0.0,
        timeout_sec: float = 60.0,
        max_retries: int = 2,
        **kwargs
    ):
        model_env = os.getenv("OLLAMA_MODEL")
        if model_env:
            model_name = model_env

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            **kwargs
        )
        self.base_url = os.getenv("OLLAMA_BASE_URL", base_url)
        self.api_key = os.getenv("OLLAMA_API_KEY", "")

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        if "chat/completions" in self.base_url:
            payload["response_format"] = {"type": "json_object"}

        data = json.dumps(payload).encode("utf-8")
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(self.base_url, data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                    res_body = response.read().decode("utf-8")
                    res_json = json.loads(res_body)

                    self.call_count += 1

                    usage = res_json.get("usage", {})
                    prompt_toks = usage.get("prompt_tokens", usage.get("prompt_eval_count", 0))
                    eval_toks = usage.get("completion_tokens", usage.get("eval_count", 0))
                    self.token_usage["prompt_tokens"] += prompt_toks
                    self.token_usage["completion_tokens"] += eval_toks
                    self.token_usage["total_tokens"] = (
                        self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"]
                    )

                    if "choices" in res_json and res_json["choices"]:
                        return res_json["choices"][0]["message"]["content"]
                    elif "message" in res_json and "content" in res_json["message"]:
                        return res_json["message"]["content"]
                    elif "response" in res_json:
                        return res_json["response"]
                    else:
                        raise ValueError(f"Unrecognized response payload structure from Ollama: {res_body[:200]}")

            except urllib.error.HTTPError as e:
                err_text = e.fp.read().decode("utf-8") if e.fp else str(e)
                logger.warning(f"Ollama API HTTPError (Attempt {attempt}/{self.max_retries}): {e.code} - {err_text}")
                last_error = f"HTTP {e.code}: {err_text}"
            except Exception as e:
                logger.warning(f"Ollama API Exception (Attempt {attempt}/{self.max_retries}): {str(e)}")
                last_error = str(e)

        raise RuntimeError(f"OllamaProvider failed after {self.max_retries} attempts: {last_error}")
