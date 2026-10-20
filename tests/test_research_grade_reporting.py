"""Comprehensive test suite for CiteGuard Academic-Grade Final Report and Synthesis Engine."""

import os
import json
import tempfile
import unittest
from fastapi.testclient import TestClient

from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.verification.schema import VerificationResult, RetrievedEvidence, AgentAssessment, AgentDecision
from src.reporting.generator import ReportGenerator, normalize_verdict
from src.ui.server import app


class TestResearchGradeReporting(unittest.TestCase):
    """Unit and integration tests for Section 4 Research Report Generation and API payload."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_supported_claims_case(self):
        """Case 1: All claims supported with evidence."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_supp",
                research_question="Does regular physical activity reduce cardiovascular disease risk?",
                domain="cardiology",
                sub_questions=["What is the effect on blood pressure?", "What is the effect on arterial stiffness?"]
            )
            v_results = [
                VerificationResult(
                    claim_id="C1",
                    claim_text="Aerobic exercise significantly lowers systolic blood pressure in hypertensive adults.",
                    final_label="SUPPORTED",
                    confidence=0.95,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="doc_01",
                            text="Meta-analysis of 40 RCTs demonstrates a 7.2 mmHg drop in systolic pressure.",
                            score=0.92,
                            source="openalex",
                            title="Exercise and Arterial Pressure",
                            metadata={
                                "paper_id": "W12345678",
                                "authors": ["Smith, J.", "Doe, A."],
                                "publication_year": 2021,
                                "section": "Abstract",
                                "passage_id": "p01",
                                "url_doi": "https://doi.org/10.1001/jama.2021.01",
                                "relationship": "supports"
                            }
                        )
                    ]
                ),
                VerificationResult(
                    claim_id="C2",
                    claim_text="Physical training reduces pulse wave velocity and arterial stiffness.",
                    final_label="SUPPORT",
                    confidence=0.91,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="doc_02",
                            text="Carotid-femoral pulse wave velocity decreased significantly after 12 weeks of training.",
                            score=0.88,
                            source="scifact",
                            title="Arterial Compliance in Athletes",
                            metadata={
                                "paper_id": "W87654321",
                                "authors": ["Taylor, R."],
                                "publication_year": 2023,
                                "section": "Results",
                                "passage_id": "p03",
                                "url_doi": "https://doi.org/10.1016/j.jacc.2023.05",
                                "relationship": "supports"
                            }
                        )
                    ]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            dash = report["dashboard"]
            self.assertEqual(dash["total_claims"], 2)
            self.assertEqual(dash["supported_count"], 2)
            self.assertEqual(dash["supported_pct"], 100.0)
            self.assertEqual(dash["partial_count"], 0)
            self.assertEqual(dash["insufficient_count"], 0)
            self.assertEqual(dash["contradicted_count"], 0)
            self.assertEqual(dash["unique_sources_cited"], 2)
            self.assertEqual(dash["evidence_coverage_pct"], 100.0)
            self.assertEqual(dash["publication_year_range"], "2021 – 2023")

            rs = report["research_synthesis"]
            self.assertEqual(rs["degree_of_support"], "HIGH_CONFIDENCE_SUPPORT")
            self.assertEqual(rs["sufficiency_status"], "SUFFICIENT")
            self.assertGreaterEqual(len(rs["strongest_conclusions"]), 2)
            self.assertIn("cardiovascular disease risk", rs["question"])

    def test_partially_supported_claims_case(self):
        """Case 2: Claims with partial support and contextual qualifications."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_partial",
                research_question="Does intermittent fasting enhance cellular autophagy in humans?",
                sub_questions=["Is autophagy induced in human muscle tissue?"]
            )
            v_results = [
                VerificationResult(
                    claim_id="C1",
                    claim_text="Intermittent fasting stimulates autophagy markers in peripheral blood mononuclear cells.",
                    final_label="PARTIALLY_SUPPORTED",
                    confidence=0.78,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="doc_p1",
                            text="Autophagy markers LC3-II increased during 24-hour fast, but findings varied widely by cohort.",
                            score=0.74,
                            title="Fasting and Autophagy",
                            metadata={"publication_year": 2020, "relationship": "partial"}
                        )
                    ]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            dash = report["dashboard"]
            self.assertEqual(dash["partial_count"], 1)
            self.assertEqual(dash["partial_pct"], 100.0)
            self.assertEqual(dash["supported_count"], 0)

            ug = report["uncertainty_and_gaps"]
            self.assertEqual(len(ug["unverified_claims"]), 1)
            self.assertEqual(ug["unverified_claims"][0]["claim_id"], "C1")
            self.assertIn("partial", ug["unverified_claims"][0]["reason"].lower())

    def test_insufficient_evidence_claims_case(self):
        """Case 3: Inconclusive/NEI claims lacking empirical data."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_nei",
                research_question="Does cold atmospheric plasma eradicate deep biofilm formation in chronic osteomyelitis?"
            )
            v_results = [
                VerificationResult(
                    claim_id="C1",
                    claim_text="Cold atmospheric plasma penetrates cortical bone to eliminate deep bacterial biofilms.",
                    final_label="INSUFFICIENT_EVIDENCE",
                    confidence=0.65,
                    evidence=[]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            dash = report["dashboard"]
            self.assertEqual(dash["insufficient_count"], 1)
            self.assertEqual(dash["insufficient_pct"], 100.0)
            self.assertEqual(dash["evidence_coverage_pct"], 0.0)
            self.assertEqual(dash["missing_evidence_count"], 1)

            rs = report["research_synthesis"]
            self.assertEqual(rs["degree_of_support"], "INSUFFICIENT_EVIDENCE")
            self.assertEqual(rs["sufficiency_status"], "INSUFFICIENT")

    def test_contradicted_claims_case(self):
        """Case 4: Claims contradicted by retrieved literature."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_con",
                research_question="Does high-dose vitamin C supplementation prevent acute viral respiratory infections?"
            )
            v_results = [
                VerificationResult(
                    claim_id="C1",
                    claim_text="High-dose vitamin C prevents the incidence of the common cold in the general population.",
                    final_label="CONTRADICTED",
                    confidence=0.94,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="doc_cochrane",
                            text="The failure of vitamin C supplementation to reduce the incidence of colds in the general population indicates that routine prophylaxis is not justified.",
                            score=0.96,
                            title="Vitamin C for Preventing and Treating the Common Cold",
                            metadata={
                                "paper_id": "COCHRANE_2013",
                                "authors": "Hemilä H, Chalker E",
                                "publication_year": 2013,
                                "relationship": "contradicts",
                                "url_doi": "https://doi.org/10.1002/14651858.CD000980.pub4"
                            }
                        )
                    ]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            dash = report["dashboard"]
            self.assertEqual(dash["contradicted_count"], 1)
            self.assertEqual(dash["contradicted_pct"], 100.0)
            self.assertEqual(dash["contradictory_sources_count"], 1)

            rs = report["research_synthesis"]
            self.assertEqual(rs["degree_of_support"], "STRONGLY_CONTRADICTED")
            self.assertEqual(rs["sufficiency_status"], "SUFFICIENT_REFUTATION")

    def test_missing_source_case(self):
        """Case 5: Claims with missing evidence handle publication year and coverage gracefully."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q_no_source", research_question="Speculative assertion without papers.")
            v_results = [
                VerificationResult(
                    claim_id="C_NONE",
                    claim_text="Speculative assertion.",
                    final_label="INSUFFICIENT_EVIDENCE",
                    confidence=0.5,
                    evidence=[]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            dash = report["dashboard"]
            self.assertEqual(dash["publication_year_range"], "Not available")
            self.assertEqual(dash["direct_vs_indirect"], "Not available")
            self.assertEqual(dash["unique_sources_cited"], 0)
            self.assertEqual(dash["evidence_coverage_pct"], 0.0)
            self.assertEqual(dash["missing_evidence_count"], 1)

    def test_conflicting_evidence_case(self):
        """Case 6: Mixed supported and contradicted claims result in contested status."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_mixed",
                research_question="Does moderate alcohol consumption reduce all-cause mortality?"
            )
            v_results = [
                VerificationResult(
                    claim_id="C1",
                    claim_text="Low-to-moderate alcohol intake exhibits a J-shaped curve with cardiovascular mortality.",
                    final_label="SUPPORTED",
                    confidence=0.82,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="d1",
                            text="Observational studies show a nadir in cardiovascular risk at 1 drink per day.",
                            score=0.85,
                            metadata={"paper_id": "P1", "publication_year": 2017, "relationship": "supports"}
                        )
                    ]
                ),
                VerificationResult(
                    claim_id="C2",
                    claim_text="Mendelian randomization studies confirm a causal protective effect on all-cause mortality.",
                    final_label="CONTRADICTED",
                    confidence=0.91,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="d2",
                            text="Genetic Mendelian randomization analysis shows no protective effect of alcohol intake on mortality.",
                            score=0.93,
                            metadata={"paper_id": "P2", "publication_year": 2022, "relationship": "contradicts"}
                        )
                    ]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                report = json.load(f)

            dash = report["dashboard"]
            self.assertEqual(dash["supported_count"], 1)
            self.assertEqual(dash["contradicted_count"], 1)
            self.assertEqual(dash["supported_pct"], 50.0)
            self.assertEqual(dash["contradicted_pct"], 50.0)
            self.assertEqual(dash["contradictory_sources_count"], 1)

            rs = report["research_synthesis"]
            self.assertEqual(rs["degree_of_support"], "CONTESTED_EVIDENCE")
            self.assertEqual(rs["sufficiency_status"], "INCONCLUSIVE_CONFLICT")

    def test_markdown_export_includes_all_sections(self):
        """Case 7: Markdown report includes all academic report sections."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(
                question_id="q_md",
                research_question="Does sleep deprivation impair hippocampal neurogenesis?",
                sub_questions=["What is the mechanism in the dentate gyrus?"]
            )
            v_results = [
                VerificationResult(
                    claim_id="C1",
                    claim_text="Chronic sleep restriction decreases proliferation of progenitor cells in dentate gyrus.",
                    final_label="SUPPORTED",
                    confidence=0.89,
                    evidence=[
                        RetrievedEvidence(
                            doc_id="doc_sd",
                            text="BrdU labeling demonstrated a 45% reduction in neurogenesis following sleep restriction.",
                            score=0.91,
                            title="Sleep and Hippocampal Proliferation",
                            metadata={
                                "paper_id": "P_NEURO",
                                "authors": "Kopp C, et al.",
                                "publication_year": 2019,
                                "url_doi": "https://doi.org/10.1523/jneurosci.2019",
                                "relationship": "supports"
                            }
                        )
                    ],
                    agent_assessments=[
                        AgentAssessment(
                            agent_name="VerifierAgent",
                            decision=AgentDecision.SUPPORT,
                            confidence=0.9,
                            rationale="BrdU evidence directly establishes reduced proliferation."
                        )
                    ]
                )
            ]

            files = rg.generate_report(rq, v_results)
            with open(files["md_path"], "r", encoding="utf-8") as f:
                md_content = f.read()

            self.assertIn("# CiteGuard Scientific Claim Verification & Research Synthesis Report", md_content)
            self.assertIn("## 1. Planner Sub-Questions Decomposition", md_content)
            self.assertIn("## 2. Executive Research Synthesis", md_content)
            self.assertIn("## 3. Research Verdict & Citation Dashboard", md_content)
            self.assertIn("## 4. Grounded Conclusions & Evidentiary Qualifications", md_content)
            self.assertIn("## 5. Claim-by-Claim In-Depth Verification & Evidence Breakdown", md_content)
            self.assertIn("## 6. Scientific Uncertainty & Literature Gaps", md_content)
            self.assertIn("## 7. Evidence-Grounded Practical Implications & Next Steps", md_content)
            self.assertIn("## 8. Methodological Limitations & Bounded Scope", md_content)

    def test_api_verify_returns_research_grade_synthesis(self):
        """Case 8: FastAPI server POST /api/verify returns dashboard and research_synthesis payloads."""
        payload = {
            "question": "What is the relationship between sleep duration and cognitive performance?",
            "provider": "mock",
            "top_k": 3,
            "mode": "hybrid"
        }
        response = self.client.post("/api/verify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertIn("dashboard", data)
        self.assertIn("research_synthesis", data)
        self.assertIn("uncertainty_and_gaps", data)

        dash = data["dashboard"]
        self.assertIn("total_claims", dash)
        self.assertIn("supported_count", dash)
        self.assertIn("supported_pct", dash)
        self.assertIn("evidence_coverage_pct", dash)
        self.assertIn("unique_sources_cited", dash)

        rs = data["research_synthesis"]
        self.assertIn("degree_of_support", rs)
        self.assertIn("sufficiency_status", rs)
        self.assertIn("main_findings", rs)
        self.assertIn("strongest_conclusions", rs)
        self.assertIn("qualifications_and_conflicts", rs)
        self.assertIn("follow_up_questions", rs)
        self.assertIn("practical_implications", rs)
        self.assertIn("recommended_next_steps", rs)


if __name__ == "__main__":
    unittest.main()
