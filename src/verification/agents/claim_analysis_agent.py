"""ClaimAnalysisAgent implementation for claim decomposition and scope evaluation."""

from typing import Optional
from src.verification.base_agent import BaseAgent
from src.verification.providers.base_provider import LLMProvider
from src.verification.prompts import CLAIM_ANALYSIS_AGENT_SYSTEM_PROMPT


class ClaimAnalysisAgent(BaseAgent):
    """Agent specialized in structural claim decomposition and required evidence scope evaluation."""

    def __init__(self, provider: Optional[LLMProvider] = None):
        super().__init__(
            name="ClaimAnalysisAgent",
            provider=provider,
            system_prompt=CLAIM_ANALYSIS_AGENT_SYSTEM_PROMPT
        )
