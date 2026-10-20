"""SourceQualityAgent implementation for metadata and source completeness evaluation."""

from typing import Optional
from src.verification.base_agent import BaseAgent
from src.verification.providers.base_provider import LLMProvider
from src.verification.prompts import SOURCE_QUALITY_AGENT_SYSTEM_PROMPT


class SourceQualityAgent(BaseAgent):
    """Agent specialized in evaluating document completeness, metadata reliability, and text noise."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        super().__init__(
            name="SourceQualityAgent",
            provider=provider,
            system_prompt=SOURCE_QUALITY_AGENT_SYSTEM_PROMPT
        )
