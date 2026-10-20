"""Factory utility for instantiating LLM Providers."""

import os
import logging
from typing import Optional, Dict, Any

from src.verification.providers.base_provider import LLMProvider
from src.verification.providers.mock_provider import MockProvider
from src.verification.providers.openai_provider import OpenAIProvider
from src.verification.providers.ollama_provider import OllamaProvider

logger = logging.getLogger(__name__)


def get_provider(
    provider_type: str = "mock",
    model_name: Optional[str] = None,
    fallback_to_mock: bool = True,
    **kwargs
) -> LLMProvider:
    """Factory function to build LLMProvider instances based on environment or configuration."""
    provider_type = (provider_type or os.getenv("LLM_PROVIDER", "mock")).lower().strip()

    if provider_type in ("openai", "gpt"):
        model = model_name or os.getenv("LLM_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            if fallback_to_mock:
                logger.warning("OPENAI_API_KEY not set. Falling back to MockProvider for offline safety.")
                return MockProvider(model_name=f"mock-{model}", **kwargs)
            else:
                raise ValueError("OPENAI_API_KEY environment variable is required for OpenAIProvider.")
        return OpenAIProvider(model_name=model, api_key=api_key, **kwargs)

    elif provider_type in ("ollama", "local"):
        model = model_name or os.getenv("OLLAMA_MODEL", "llama3.2")
        return OllamaProvider(model_name=model, **kwargs)

    elif provider_type == "mock":
        model = model_name or "mock-model"
        return MockProvider(model_name=model, **kwargs)

    else:
        if fallback_to_mock:
            logger.warning(f"Unknown provider_type '{provider_type}'. Falling back to MockProvider.")
            return MockProvider(model_name=model_name or "mock-model", **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider type: '{provider_type}'")

