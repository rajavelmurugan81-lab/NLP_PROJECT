"""Regression tests for CiteGuard Verdict Normalization and Synthesis Flow."""

import unittest
import json
import os
import tempfile
from fastapi.testclient import TestClient

from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.reporting.generator import ReportGenerator, normalize_verdict
from src.verification.schema import VerificationResult, RetrievedEvidence
from src.ui.server import app


class TestVerdictSynthesisFlow(unittest.TestCase):
    """Regression unit test suite for claim verdict data flow and report synthesis."""

    def setUp(self):
        self.client = TestClient(app)

    def test_verdict_counts_are_nonzero_when_verified_claims_exist(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_test_1",
                research_question="What are the effects of sleep deprivation on memory consolidation?",
                generated_claims=[
                    GeneratedClaim(claim_id="c1", question_id="q_test_1", claim_text="Sleep loss impairs spatial memory."),
                    GeneratedClaim(claim_id="c2", question_id="q_test_1", claim_text="REM sleep promotes synaptic plasticity.")
                ]
            )
            v_results = [
                VerificationResult(claim_id="c1", claim_text="Sleep loss impairs spatial memory.", final_label="SUPPORTED", confidence=0.92),
                VerificationResult(claim_id="c2", claim_text="REM sleep promotes synaptic plasticity.", final_label="SUPPORT", confidence=0.88)
            ]
            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)
            
            self.assertEqual(report["verification"]["supported"], 2)
            self.assertEqual(report["synthesis"]["supported_count"], 2)
            self.assertIn("2 supported", report["overall_synthesis"])

    def test_support_and_supported_are_normalized(self):
        self.assertEqual(normalize_verdict("SUPPORT"), "SUPPORTED")
        self.assertEqual(normalize_verdict("SUPPORTED"), "SUPPORTED")
        self.assertEqual(normalize_verdict("SUPPORTS"), "SUPPORTED")

    def test_partial_verdict_is_normalized(self):
        self.assertEqual(normalize_verdict("PARTIAL"), "PARTIALLY_SUPPORTED")
        self.assertEqual(normalize_verdict("PARTIALLY_SUPPORTED"), "PARTIALLY_SUPPORTED")
        self.assertEqual(normalize_verdict("PARTIALLY SUPPORTED"), "PARTIALLY_SUPPORTED")

    def test_insufficient_verdict_is_normalized(self):
        self.assertEqual(normalize_verdict("NEI"), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(normalize_verdict("INSUFFICIENT"), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(normalize_verdict("INSUFFICIENT_EVIDENCE"), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(normalize_verdict("NEUTRAL"), "INSUFFICIENT_EVIDENCE")
        self.assertEqual(normalize_verdict("NO_EVIDENCE"), "INSUFFICIENT_EVIDENCE")

    def test_contradicted_verdict_is_normalized(self):
        self.assertEqual(normalize_verdict("REFUTE"), "CONTRADICTED")
        self.assertEqual(normalize_verdict("REFUTED"), "CONTRADICTED")
        self.assertEqual(normalize_verdict("CONTRADICT"), "CONTRADICTED")
        self.assertEqual(normalize_verdict("CONTRADICTED"), "CONTRADICTED")

    def test_synthesis_counts_match_verified_claims(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_test_2",
                research_question="Does miRNA regulation impact cancer cell invasion?",
                generated_claims=[]
            )
            v_results = [
                VerificationResult(claim_id="c1", claim_text="Claim 1", final_label="SUPPORTED", confidence=0.9),
                VerificationResult(claim_id="c2", claim_text="Claim 2", final_label="PARTIALLY_SUPPORTED", confidence=0.8),
                VerificationResult(claim_id="c3", claim_text="Claim 3", final_label="INSUFFICIENT_EVIDENCE", confidence=0.7)
            ]
            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            synth = report["synthesis"]
            self.assertEqual(synth["supported_count"], 1)
            self.assertEqual(synth["partial_count"], 1)
            self.assertEqual(synth["insufficient_count"], 1)
            self.assertEqual(synth["contradicted_count"], 0)
            self.assertIn("1 supported, 1 partially supported, 1 insufficient evidence, 0 contradicted", report["overall_synthesis"])

    def test_contract_anomalies_are_independent_of_verdict_counts(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q_test_3", research_question="Test Q", generated_claims=[])
            v_results = [
                VerificationResult(
                    claim_id="c1",
                    claim_text="Supported claim text",
                    final_label="SUPPORTED",
                    confidence=0.95,
                    failure_signature=None # Clean contract anomaly count = 0
                )
            ]
            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            self.assertEqual(report["verification"]["supported"], 1)
            self.assertIsNone(report["claims"][0]["failure_signature"])

    def test_api_returns_verified_claims(self):
        response = self.client.post(
            "/api/verify",
            json={
                "question": "What are the effects of sleep deprivation on memory consolidation?",
                "mode": "openalex_hybrid",
                "top_k": 3
            }
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertIn("claims", data)
        self.assertIn("verification", data)
        self.assertIn("synthesis", data)
        self.assertGreater(len(data["claims"]), 0)
        self.assertIn("supported", data["verification"])

    def test_frontend_uses_final_verification_counts(self):
        static_app_js = os.path.join(os.path.dirname(__file__), "..", "src", "ui", "static", "app.js")
        self.assertTrue(os.path.exists(static_app_js))
        with open(static_app_js, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("normalizeVerdict", content)
        self.assertIn("renderResults", content)
        self.assertIn("limitationsList", content)


if __name__ == "__main__":
    unittest.main()
