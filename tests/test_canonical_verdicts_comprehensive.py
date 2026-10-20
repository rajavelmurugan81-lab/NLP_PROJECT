"""Comprehensive test suite for deterministic canonical verification verdicts in CiteGuard."""

import unittest
import tempfile
import json

from src.verification.schema import (
    VerificationRequest,
    RetrievedEvidence,
    AgentDecision,
    VerificationResult
)
from src.verification.orchestrator import MultiAgentOrchestrator
from src.verification.providers.mock_provider import MockProvider
from src.verification.evidence_mapper import EvidenceMapper, classify_relationship
from src.reporting.generator import ReportGenerator, normalize_verdict
from src.claim_generation.schema import ResearchQuestion, GeneratedClaim


class TestCanonicalVerdictsComprehensive(unittest.TestCase):
    """Test suite ensuring deterministic verification outcomes for all 4 canonical verdicts."""

    def test_deterministic_partially_supported_evidence_mapping(self):
        """Test evidence mapping and classification for a multi-aspect claim with partial support."""
        claim_text = "Sleep deprivation reduces executive function and alters prefrontal cortex theta activity."
        
        # Passage 1: Supports executive function aspect modestly under certain conditions
        passage_1 = "Controlled sleep deprivation modestly reduces executive function performance under certain conditions."
        rel1, score1 = classify_relationship(claim_text, passage_1)
        self.assertEqual(rel1, "PARTIALLY_SUPPORTS")

        # Passage 2: Inconclusive/preliminary evidence for prefrontal cortex theta activity aspect
        passage_2 = "Preliminary electroencephalography studies provide insufficient evidence regarding prefrontal cortex theta activity during sleep loss."
        rel2, score2 = classify_relationship(claim_text, passage_2)
        self.assertEqual(rel2, "PARTIALLY_SUPPORTS")

        # Map passages using EvidenceMapper
        mapper = EvidenceMapper()
        passages = [
            {"paper_id": "doc_p1", "passage_id": "doc_p1_01", "text": passage_1, "title": "Sleep and Cognition Study", "relevance_score": 0.85},
            {"paper_id": "doc_p2", "passage_id": "doc_p2_01", "text": passage_2, "title": "EEG Analysis Study", "relevance_score": 0.80}
        ]
        links = mapper.map_claim_to_passages("claim_part_1", claim_text, passages)
        self.assertEqual(len(links), 2)
        self.assertTrue(all(l.relationship == "PARTIALLY_SUPPORTS" for l in links))

    def test_deterministic_partially_supported_verdict(self):
        """Test multi-agent verification pipeline producing a PARTIALLY_SUPPORTED verdict."""
        claim_text = "Sleep deprivation reduces executive function and alters prefrontal cortex theta activity."
        
        pre_evidence = [
            RetrievedEvidence(
                doc_id="doc_p1",
                text="Controlled sleep deprivation modestly reduces executive function performance under certain conditions.",
                score=0.75,
                rank=1,
                source="openalex",
                title="Sleep and Executive Function Study"
            ),
            RetrievedEvidence(
                doc_id="doc_p2",
                text="Preliminary electroencephalography studies provide insufficient evidence regarding prefrontal cortex theta activity during sleep loss.",
                score=0.70,
                rank=2,
                source="openalex",
                title="Prefrontal Cortex EEG Study"
            )
        ]

        provider = MockProvider(default_decision="PARTIAL", default_confidence=0.82)
        orchestrator = MultiAgentOrchestrator(provider=provider)

        req = VerificationRequest(
            claim_id="claim_partial_001",
            claim_text=claim_text,
            retrieval_mode="openalex_hybrid"
        )

        res = orchestrator.verify(request=req, pre_retrieved_evidence=pre_evidence)
        
        # Verify raw decision and normalized canonical verdict
        self.assertEqual(res.final_label, "PARTIAL")
        normalized = normalize_verdict(res.final_label)
        self.assertEqual(normalized, "PARTIALLY_SUPPORTED")

    def test_deterministic_supported_verdict(self):
        """Test multi-agent verification pipeline producing a SUPPORTED verdict."""
        claim_text = "Increasing model size improves accuracy in large language models."
        pre_evidence = [
            RetrievedEvidence(
                doc_id="doc_sup_1",
                text="Empirical evaluation across transformer architectures confirms that increasing model size improves accuracy in large language models.",
                score=0.92,
                rank=1,
                source="openalex",
                title="Scaling Laws for Neural Language Models"
            )
        ]

        provider = MockProvider(default_decision="SUPPORT", default_confidence=0.90)
        orchestrator = MultiAgentOrchestrator(provider=provider)

        req = VerificationRequest(claim_id="claim_sup_001", claim_text=claim_text)
        res = orchestrator.verify(request=req, pre_retrieved_evidence=pre_evidence)

        self.assertEqual(res.final_label, "SUPPORT")
        self.assertEqual(normalize_verdict(res.final_label), "SUPPORTED")

    def test_deterministic_insufficient_evidence_verdict(self):
        """Test verification pipeline producing INSUFFICIENT_EVIDENCE when no relevant evidence exists."""
        claim_text = "Quantum entanglement directly alters cellular ATP metabolism in human cardiac myocytes."
        
        # Empty pre-retrieved evidence list
        provider = MockProvider(default_decision="NEI", default_confidence=0.90)
        orchestrator = MultiAgentOrchestrator(provider=provider)

        req = VerificationRequest(claim_id="claim_nei_001", claim_text=claim_text)
        res = orchestrator.verify(request=req, pre_retrieved_evidence=[])

        self.assertEqual(res.final_label, "NEI")
        self.assertEqual(normalize_verdict(res.final_label), "INSUFFICIENT_EVIDENCE")

    def test_deterministic_contradicted_verdict(self):
        """Test verification pipeline producing CONTRADICTED verdict when evidence directly conflicts."""
        claim_text = "Sleep deprivation significantly improves long-term memory consolidation."
        pre_evidence = [
            RetrievedEvidence(
                doc_id="doc_ref_1",
                text="Controlled experiments demonstrate that sleep deprivation does not improve memory consolidation and in fact severely degrades memory retention.",
                score=0.88,
                rank=1,
                source="openalex",
                title="Memory Degradation Under Sleep Loss"
            )
        ]

        provider = MockProvider(default_decision="REFUTE", default_confidence=0.88)
        orchestrator = MultiAgentOrchestrator(provider=provider)

        req = VerificationRequest(claim_id="claim_ref_001", claim_text=claim_text)
        res = orchestrator.verify(request=req, pre_retrieved_evidence=pre_evidence)

        self.assertEqual(res.final_label, "REFUTE")
        self.assertEqual(normalize_verdict(res.final_label), "CONTRADICTED")

    def test_report_synthesis_distribution_all_verdicts(self):
        """Test report generator accurately synthesizing all 4 canonical verdicts in summary stats."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_all_verdicts",
                research_question="Multidisciplinary evaluation question",
                generated_claims=[]
            )
            v_results = [
                VerificationResult(claim_id="c1", claim_text="Claim 1", final_label="SUPPORTED", confidence=0.90),
                VerificationResult(claim_id="c2", claim_text="Claim 2", final_label="PARTIALLY_SUPPORTED", confidence=0.80),
                VerificationResult(claim_id="c3", claim_text="Claim 3", final_label="INSUFFICIENT_EVIDENCE", confidence=0.70),
                VerificationResult(claim_id="c4", claim_text="Claim 4", final_label="CONTRADICTED", confidence=0.85)
            ]
            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                data = json.load(f)

            summary = data["verification"]
            self.assertEqual(summary["supported"], 1)
            self.assertEqual(summary["partially_supported"], 1)
            self.assertEqual(summary["insufficient_evidence"], 1)
            self.assertEqual(summary["contradicted"], 1)
            
            synthesis_str = data["overall_synthesis"]
            self.assertIn("1 supported", synthesis_str)
            self.assertIn("1 partially supported", synthesis_str)
            self.assertIn("1 insufficient evidence", synthesis_str)
            self.assertIn("1 contradicted", synthesis_str)


if __name__ == "__main__":
    unittest.main()
