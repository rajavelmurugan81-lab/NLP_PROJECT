"""Unit tests for LLM provider abstraction and MockProvider."""

import pytest
from src.verification.providers.mock_provider import MockProvider
from src.verification.providers.openai_provider import OpenAIProvider
from src.verification.providers.factory import get_provider


def test_mock_provider_defaults():
    provider = MockProvider(default_decision="SUPPORT", default_confidence=0.9)
    resp = provider.generate("Test prompt", system_prompt="You are an evidence agent.")
    
    assert provider.call_count == 1
    assert "SUPPORT" in resp
    
    structured = provider.generate_structured("Test prompt", system_prompt="You are evidence agent.")
    assert structured["decision"] == "SUPPORT"
    assert structured["confidence"] == 0.9


def test_mock_provider_rule_function():
    def custom_rule(prompt, sys_prompt):
        if "refute" in prompt.lower():
            return '{"decision": "REFUTE", "confidence": 0.95}'
        return None

    provider = MockProvider(rule_fn=custom_rule)
    res_support = provider.generate_structured("Please check this claim")
    assert res_support["decision"] == "SUPPORT"

    res_refute = provider.generate_structured("Please refute this claim")
    assert res_refute["decision"] == "REFUTE"


def test_provider_factory_fallback():
    # Calling get_provider("openai") without API key should gracefully fall back to MockProvider
    provider = get_provider(provider_type="openai", fallback_to_mock=True)
    assert isinstance(provider, MockProvider)
    assert "mock" in provider.model_name


def test_openai_provider_missing_key():
    provider = OpenAIProvider(api_key="")
    with pytest.raises(ValueError, match="API key not found"):
        provider.generate("Test prompt")
