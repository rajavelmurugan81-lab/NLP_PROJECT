"""OpenAI LLM Provider implementation for CiteGuard."""

import os
import json
import logging
from typing import Dict, Any, Optional
import urllib.request
import urllib.error

from src.verification.providers.base_provider import LLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI API Provider supporting HTTP request execution with fallback/retry handling."""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        timeout_sec: float = 30.0,
        max_retries: int = 2,
        base_url: str = "https://api.openai.com/v1/chat/completions",
        **kwargs
    ):
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            **kwargs
        )
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", base_url)

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"}
        }

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
                    self.token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    self.token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    self.token_usage["total_tokens"] = (
                        self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"]
                    )

                    choices = res_json.get("choices", [])
                    if choices:
                        return choices[0]["message"]["content"]
                    else:
                        raise ValueError("Empty choices list received from OpenAI API.")

            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8") if e.fp else str(e)
                logger.warning(f"OpenAI API HTTPError (Attempt {attempt}/{self.max_retries}): {e.code} - {err_text}")
                last_error = f"HTTP {e.code}: {err_text}"
            except Exception as e:
                logger.warning(f"OpenAI API Exception (Attempt {attempt}/{self.max_retries}): {str(e)}")
                last_error = str(e)

        raise RuntimeError(f"OpenAIProvider failed after {self.max_retries} attempts: {last_error}")
