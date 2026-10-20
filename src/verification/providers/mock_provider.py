"""Mock LLM Provider for deterministic offline testing."""

from typing import Dict, Any, Optional, List, Callable
import json
from src.verification.providers.base_provider import LLMProvider


class MockProvider(LLMProvider):
    """Deterministic Mock LLM Provider for offline testing without external API dependency."""

    def __init__(
        self,
        model_name: str = "mock-gpt-4",
        default_decision: str = "SUPPORT",
        default_confidence: float = 0.85,
        custom_responses: Optional[Dict[str, str]] = None,
        rule_fn: Optional[Callable[[str, Optional[str]], str]] = None,
        **kwargs
    ):
        super().__init__(model_name=model_name, **kwargs)
        self.default_decision = default_decision
        self.default_confidence = default_confidence
        self.custom_responses = custom_responses or {}
        self.rule_fn = rule_fn

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_count += 1
        
        # Estimate mock token count
        prompt_tokens = len(prompt.split()) + (len(system_prompt.split()) if system_prompt else 0)
        
        # Check rule function if registered
        if self.rule_fn:
            resp = self.rule_fn(prompt, system_prompt)
            if resp:
                self.token_usage["prompt_tokens"] += prompt_tokens
                self.token_usage["completion_tokens"] += len(resp.split())
                self.token_usage["total_tokens"] = self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"]
                return resp

        # Check exact or substring match in custom_responses
        for key, val in self.custom_responses.items():
            if key in prompt or (system_prompt and key in system_prompt):
                self.token_usage["prompt_tokens"] += prompt_tokens
                self.token_usage["completion_tokens"] += len(val.split())
                self.token_usage["total_tokens"] = self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"]
                return val

        # Infer response structure based on agent role in system prompt or prompt keywords
        prompt_lower = (prompt + " " + (system_prompt or "")).lower()

        decision = self.default_decision

        if "debate" in prompt_lower or "resolver" in prompt_lower or "resolution" in prompt_lower:
            refute_mentions = prompt.count("Decision: REFUTE")
            partial_mentions = prompt.count("Decision: PARTIAL")
            nei_mentions = prompt.count("Decision: NEI")
            support_mentions = prompt.count("Decision: SUPPORT")

            if refute_mentions >= 2 or (refute_mentions > 0 and support_mentions <= 1):
                d_decision = "REFUTE"
            elif partial_mentions >= 2:
                d_decision = "PARTIAL"
            elif nei_mentions >= 2 and support_mentions < 3:
                d_decision = "PARTIAL" if partial_mentions > 0 else "NEI"
            elif (nei_mentions + partial_mentions) > support_mentions:
                d_decision = "PARTIAL" if partial_mentions > 0 else "NEI"
            else:
                d_decision = decision

            output_dict = {
                "final_decision": d_decision,
                "consensus_reached": True,
                "resolution_rationale": f"Mock Debate Resolver synthesized evidence and agent assessments to conclude {d_decision}.",
                "disagreement_reasons": []
            }
        elif "evidence agent" in prompt_lower or "evidence_agent" in prompt_lower:
            output_dict = {
                "decision": decision,
                "confidence": self.default_confidence,
                "supporting_evidence_ids": ["doc_001"] if decision == "SUPPORT" else [],
                "contradicting_evidence_ids": ["doc_001"] if decision == "REFUTE" else [],
                "rationale": f"Mock Evidence Agent evaluated claim as {decision} based on passage similarity.",
                "uncertainty": round(1.0 - self.default_confidence, 2)
            }
        elif "claim analysis agent" in prompt_lower or "claim_analysis_agent" in prompt_lower:
            output_dict = {
                "decision": decision,
                "confidence": self.default_confidence,
                "supporting_evidence_ids": [],
                "contradicting_evidence_ids": [],
                "rationale": "Mock Claim Analysis Agent decomposed claim into core atomic components.",
                "uncertainty": 0.1
            }
        elif "source quality agent" in prompt_lower or "source_quality_agent" in prompt_lower:
            output_dict = {
                "decision": "SUPPORT" if decision != "NEI" else "NEI",
                "confidence": self.default_confidence,
                "supporting_evidence_ids": ["doc_001"],
                "contradicting_evidence_ids": [],
                "rationale": "Mock Source Quality Agent verified metadata completeness and paper indexing.",
                "uncertainty": round(1.0 - self.default_confidence, 2)
            }
        elif "verification agent" in prompt_lower or "verification synthesis agent" in prompt_lower or "verification_agent" in prompt_lower:
            output_dict = {
                "decision": decision,
                "confidence": self.default_confidence,
                "supporting_evidence_ids": ["doc_001"] if decision == "SUPPORT" else [],
                "contradicting_evidence_ids": ["doc_001"] if decision == "REFUTE" else [],
                "rationale": f"Mock Verification Agent rendered decision {decision}.",
                "uncertainty": round(1.0 - self.default_confidence, 2)
            }
        elif "skeptic agent" in prompt_lower or "skeptic_agent" in prompt_lower or "adversarial skeptic" in prompt_lower:
            is_refute = decision == "REFUTE"
            output_dict = {
                "decision": "REFUTE" if is_refute else ("SUPPORT" if decision == "SUPPORT" else "NEI"),
                "confidence": self.default_confidence,
                "objection_level": "CONFIRMED_FAILURE" if is_refute else ("NO_OBJECTION" if decision == "SUPPORT" else "POTENTIAL_FAILURE"),
                "primary_failure_code": "F7" if is_refute else None,
                "failure_candidates": ["F7"] if is_refute else [],
                "objections": ["Contradictory evidence detected in literature."] if is_refute else [],
                "supporting_evidence_ids": ["doc_001"] if decision == "SUPPORT" else [],
                "contradicting_evidence_ids": ["doc_001"] if is_refute else [],
                "rationale": f"Mock Skeptic Agent rendered adversarial review for {decision}.",
                "recommended_action": "FORCE_CONTRADICTED_OR_PARTIAL" if is_refute else "ACCEPT",
                "uncertainty": round(1.0 - self.default_confidence, 2)
            }
        else:
            output_dict = {
                "decision": decision,
                "confidence": self.default_confidence,
                "supporting_evidence_ids": ["doc_001"] if decision == "SUPPORT" else [],
                "contradicting_evidence_ids": ["doc_001"] if decision == "REFUTE" else [],
                "rationale": f"Mock Verification Provider rendered decision {decision}.",
                "uncertainty": round(1.0 - self.default_confidence, 2)
            }

        resp_text = json.dumps(output_dict, indent=2)
        completion_tokens = len(resp_text.split())
        self.token_usage["prompt_tokens"] += prompt_tokens
        self.token_usage["completion_tokens"] += completion_tokens
        self.token_usage["total_tokens"] = self.token_usage["prompt_tokens"] + self.token_usage["completion_tokens"]
        return resp_text
