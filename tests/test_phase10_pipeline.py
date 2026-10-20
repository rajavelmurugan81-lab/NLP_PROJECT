"""Test suite for Phase 10: Research Question to Automatically Generated Claims to Verified Scientific Report Pipeline."""

import os
import json
import tempfile
import unittest
from typing import List

from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.claim_generation.validator import (
    validate_research_question,
    validate_generated_claim,
    deduplicate_claims
)
from src.claim_generation.generator import ClaimGenerator
from src.reporting.generator import ReportGenerator
from src.research_pipeline import ResearchPipeline
from src.verification.schema import VerificationResult, RetrievedEvidence, ExecutionMetrics


class TestPhase10Pipeline(unittest.TestCase):
    """Phase 10 Comprehensive Test Suite."""

    def test_1_validate_research_question(self):
        # Valid question
        is_valid, norm, err = validate_research_question("What is the relationship between sleep duration and cognitive performance?")
        self.assertTrue(is_valid)
        self.assertEqual(err, "")
        self.assertIn("sleep duration", norm)

        # Invalid short question
        is_valid, norm, err = validate_research_question("Sleep?")
        self.assertFalse(is_valid)
        self.assertIn("too short", err)

        # Invalid empty question
        is_valid, norm, err = validate_research_question("")
        self.assertFalse(is_valid)

    def test_2_generated_claim_schema(self):
        c = GeneratedClaim(claim_id="c1", question_id="q1", claim_text="Sleep deprivation reduces attention.", specificity_score=0.9)
        d = c.to_dict()
        self.assertEqual(d["claim_id"], "c1")
        self.assertEqual(d["claim_text"], "Sleep deprivation reduces attention.")
        
        c_restored = GeneratedClaim.from_dict(d)
        self.assertEqual(c_restored.claim_id, "c1")

    def test_3_research_question_schema(self):
        rq = ResearchQuestion(
            question_id="q100",
            research_question="Does miR-21 affect apoptosis?",
            generated_claims=[
                GeneratedClaim(claim_id="c101", question_id="q100", claim_text="MicroRNA-21 suppresses cell apoptosis.")
            ]
        )
        d = rq.to_dict()
        self.assertEqual(d["question_id"], "q100")
        self.assertEqual(len(d["generated_claims"]), 1)

    def test_4_deduplicate_claims(self):
        claims = [
            GeneratedClaim(claim_id="c1", question_id="q1", claim_text="Sleep deprivation reduces executive function."),
            GeneratedClaim(claim_id="c2", question_id="q1", claim_text="sleep deprivation reduces executive function"),  # Duplicate
            GeneratedClaim(claim_id="c3", question_id="q1", claim_text="Chronic sleep loss alters prefrontal cortex activity.")
        ]
        unique = deduplicate_claims(claims)
        self.assertEqual(len(unique), 2)
        self.assertEqual(unique[0].claim_id, "c1")
        self.assertEqual(unique[1].claim_id, "c3")

    def test_5_validate_generated_claim_vague_terms(self):
        c_vague = GeneratedClaim(claim_id="c1", question_id="q1", claim_text="Sleep is good for health stuff.")
        is_valid, err = validate_generated_claim(c_vague)
        self.assertFalse(is_valid)
        self.assertIn("vague", err)

    def test_6_validate_generated_claim_fabricated_citations(self):
        c_fake_cite = GeneratedClaim(claim_id="c1", question_id="q1", claim_text="MicroRNA-21 inhibits apoptosis in cancer (Smith et al., 2020).")
        is_valid, err = validate_generated_claim(c_fake_cite)
        self.assertFalse(is_valid)
        self.assertIn("citation", err)

    def test_7_claim_generator_mock_mode(self):
        gen = ClaimGenerator(provider_type="mock")
        rq = gen.generate_claims("What is the relationship between sleep duration and cognitive performance?")
        self.assertGreaterEqual(len(rq.generated_claims), 3)
        self.assertTrue(all(c.is_valid for c in rq.generated_claims))
        self.assertTrue(all(len(c.claim_text) >= 15 for c in rq.generated_claims))

    def test_8_claim_generator_provider_fallback(self):
        # Attempt unknown provider, should fallback to mock without crashing
        gen = ClaimGenerator(provider_type="unknown_invalid_provider")
        rq = gen.generate_claims("How does microRNA-21 regulate apoptosis in cancer cells?")
        self.assertGreaterEqual(len(rq.generated_claims), 3)

    def test_9_research_pipeline_retrieval_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = ResearchPipeline(provider_type="mock", output_dir=tmp_dir)
            res = pipeline.run_pipeline("What is the relationship between sleep and memory?", verbose=False)
            self.assertIn("verification_results", res)
            for vr in res["verification_results"]:
                self.assertIsNotNone(vr.evidence)
                self.assertGreater(len(vr.evidence), 0)
                self.assertGreater(vr.evidence[0].score, 0.0)

    def test_10_research_pipeline_verification_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = ResearchPipeline(provider_type="mock", output_dir=tmp_dir)
            res = pipeline.run_pipeline("How does microRNA-21 affect carcinoma cell apoptosis?", verbose=False)
            results = res["verification_results"]
            self.assertEqual(len(results), len(res["research_question"].generated_claims))
            for r in results:
                self.assertIn(r.final_label, ["SUPPORT", "SUPPORTED", "PARTIALLY_SUPPORTED", "REFUTE", "REFUTED", "NEUTRAL_INSUFFICIENT_EVIDENCE", "NEI", "INSUFFICIENT_EVIDENCE", "CONTRADICTED"])

    def test_11_research_pipeline_ledger_integration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = ResearchPipeline(provider_type="mock", output_dir=tmp_dir)
            res = pipeline.run_pipeline("Does CRISPR-Cas9 targeted modification reduce off-target cleavage?", verbose=False)
            self.assertTrue(os.path.exists(pipeline.verification_pipeline.ledger_path))

    def test_12_report_generator_verdict_mapping(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q1", research_question="Does sleep affect attention?", generated_claims=[])
            v_res = [
                VerificationResult(claim_id="c1", claim_text="Sleep reduces error rates.", final_label="SUPPORTED", confidence=0.9),
                VerificationResult(claim_id="c2", claim_text="Sleep alters brain activity.", final_label="NEI", confidence=0.8)
            ]
            files = rg.generate_report(rq, v_res)
            self.assertTrue(os.path.exists(files["json_path"]))
            self.assertTrue(os.path.exists(files["md_path"]))

    def test_13_report_generator_confidence_calculation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q1", research_question="Test Q", generated_claims=[])
            v_res = [
                VerificationResult(claim_id="c1", claim_text="Claim A", final_label="SUPPORTED", confidence=0.8),
                VerificationResult(claim_id="c2", claim_text="Claim B", final_label="SUPPORTED", confidence=1.0)
            ]
            rg.generate_report(rq, v_res)
            with open(os.path.join(tmp_dir, "question_q1.json"), "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["average_confidence"], 0.9)

    def test_14_report_generator_markdown_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q_md", research_question="What is the role of miR-21?", generated_claims=[])
            v_res = [
                VerificationResult(
                    claim_id="c_md_1",
                    claim_text="miR-21 regulates cell apoptosis.",
                    final_label="SUPPORTED",
                    confidence=0.95,
                    evidence=[RetrievedEvidence(doc_id="d100", text="Evidence text sample.", score=0.9, rank=1, title="Sample Study")]
                )
            ]
            files = rg.generate_report(rq, v_res)
            with open(files["md_path"], "r", encoding="utf-8") as f:
                md_text = f.read()
            self.assertIn("# Scientific Claim Verification & Research Report", md_text)
            self.assertIn("What is the role of miR-21?", md_text)
            self.assertIn("SUPPORTED", md_text)
            self.assertIn("Sample Study", md_text)

    def test_15_report_generator_json_export(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q_json", research_question="Test question for JSON export", generated_claims=[])
            v_res = [
                VerificationResult(claim_id="c_j1", claim_text="Generated claim text", final_label="SUPPORTED", confidence=0.88)
            ]
            files = rg.generate_report(rq, v_res)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["question_id"], "q_json")
            self.assertEqual(len(data["claims"]), 1)
            self.assertEqual(data["claims"][0]["claim_id"], "c_j1")

    def test_16_research_pipeline_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pipeline = ResearchPipeline(provider_type="mock", output_dir=tmp_dir)
            res = pipeline.run_pipeline("What is the relationship between sleep duration and cognitive performance?", verbose=False)
            
            self.assertIn("research_question", res)
            self.assertIn("verification_results", res)
            self.assertIn("report_files", res)
            self.assertTrue(os.path.exists(res["report_files"]["json_path"]))
            self.assertTrue(os.path.exists(res["report_files"]["md_path"]))


if __name__ == "__main__":
    unittest.main()
