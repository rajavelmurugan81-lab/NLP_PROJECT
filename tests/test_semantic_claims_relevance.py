"""Regression unit tests for semantic claim generation and evidence relevance filtering."""

import unittest
import json
import os
import tempfile

from src.claim_generation.generator import ClaimGenerator, parse_scientific_question
from src.claim_generation.validator import validate_generated_claim
from src.retrieval.openalex import clean_query_for_openalex, extract_authors_from_work
from src.verification.evidence_mapper import classify_relationship, EvidenceMapper
from src.reporting.generator import ReportGenerator
from src.claim_generation.schema import ResearchQuestion, GeneratedClaim
from src.verification.schema import VerificationResult


class TestSemanticClaimsRelevance(unittest.TestCase):
    """Test suite covering semantic X/Y extraction, relevance filtering, and author metadata."""

    def test_correct_x_extraction(self):
        p1 = parse_scientific_question("What are the effects of sleep deprivation on memory consolidation?")
        self.assertEqual(p1["var_x"].lower(), "sleep deprivation")

        p2 = parse_scientific_question("How does model size affect the accuracy and computational cost of large language models?")
        self.assertEqual(p2["var_x"].lower(), "model size")

        p3 = parse_scientific_question("Does miRNA expression play a role in cancer progression?")
        self.assertEqual(p3["var_x"].lower(), "mirna expression")

        p4 = parse_scientific_question("How does CRISPR-Cas9 achieve targeted DNA cleavage?")
        self.assertEqual(p4["var_x"].lower(), "crispr-cas9")

    def test_correct_y_extraction(self):
        p1 = parse_scientific_question("What are the effects of sleep deprivation on memory consolidation?")
        self.assertEqual(p1["var_y"].lower(), "memory consolidation")

        p2 = parse_scientific_question("Does miRNA expression play a role in cancer progression?")
        self.assertEqual(p2["var_y"].lower(), "cancer progression")

        p3 = parse_scientific_question("How does CRISPR-Cas9 achieve targeted DNA cleavage?")
        self.assertEqual(p3["var_y"].lower(), "targeted dna cleavage")

    def test_claim_semantic_validity(self):
        gen = ClaimGenerator(provider_type="mock")
        rq = gen.generate_claims("What are the effects of sleep deprivation on memory consolidation?")
        self.assertEqual(len(rq.generated_claims), 3)
        for c in rq.generated_claims:
            self.assertTrue(c.is_valid)
            self.assertNotIn("effects sleep directly affect deprivation", c.claim_text.lower())
            self.assertIn("sleep", c.claim_text.lower())
            self.assertIn("memory", c.claim_text.lower())

    def test_domain_leakage_prevention(self):
        c_bad = GeneratedClaim(
            claim_id="c1",
            question_id="q1",
            claim_text="Model size modulates prefrontal cortex cell apoptosis and tumor suppressor pathways."
        )
        is_valid, err = validate_generated_claim(c_bad, research_question="How does model size affect accuracy of LLMs?")
        self.assertFalse(is_valid)
        self.assertIn("unrelated biomedical terminology", err)

    def test_unrelated_evidence_rejection(self):
        claim = "Increasing model size improves accuracy in large language models."
        unrelated_passage = "In clinical trials of ovarian carcinoma patients, paclitaxel reduced tumor volume significantly."
        rel_label, score = classify_relationship(claim, unrelated_passage)
        self.assertEqual(rel_label, "IRRELEVANT")

    def test_openalex_query_correctness(self):
        q1 = "What are the effects of sleep deprivation on memory consolidation?"
        clean1 = clean_query_for_openalex(q1)
        self.assertIn("sleep deprivation", clean1.lower())
        self.assertIn("memory consolidation", clean1.lower())

        q2 = "How does model size affect the accuracy and computational cost of large language models?"
        clean2 = clean_query_for_openalex(q2)
        self.assertIn("model size", clean2.lower())
        self.assertIn("language models", clean2.lower())

    def test_evidence_relevance_threshold(self):
        em = EvidenceMapper()
        claim = "CRISPR-Cas9 enables targeted DNA cleavage through guide RNA-directed site recognition."
        passages = [
            {
                "paper_id": "p1",
                "passage_id": "p1_01",
                "text": "CRISPR-Cas9 induces targeted double-strand DNA cleavage guided by single-guide RNA sequences.",
                "title": "Genome editing using CRISPR-Cas9",
                "source": "openalex",
                "relevance_score": 0.9
            },
            {
                "paper_id": "p2",
                "passage_id": "p2_01",
                "text": "Deforestation in Amazon basin affects local rainfall patterns and temperature.",
                "title": "Amazon basin climate studies",
                "source": "openalex",
                "relevance_score": 0.8
            }
        ]
        links = em.map_claim_to_passages("c1", claim, passages)
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0].paper_id, "p1")
        self.assertEqual(links[0].relationship, "SUPPORTS")

    def test_insufficient_evidence_verdict(self):
        em = EvidenceMapper()
        links = [] # Zero relevant evidence links mapped
        is_suff, metrics = em.evaluate_sufficiency(links)
        self.assertFalse(is_suff)
        self.assertEqual(metrics["total_passages"], 0)

    def test_real_author_extraction(self):
        mock_work = {
            "id": "https://openalex.org/W123456",
            "display_name": "Attention Is All You Need",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
                {"author": {"display_name": "Noam Shazeer"}}
            ]
        }
        authors = extract_authors_from_work(mock_work)
        self.assertEqual(authors, ["Ashish Vaswani", "Noam Shazeer"])

    def test_synthesis_counting(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            rg = ReportGenerator(output_dir=tmp_dir)
            rq = ResearchQuestion(question_id="q1", research_question="Test Q", generated_claims=[])
            v_res = [
                VerificationResult(claim_id="c1", claim_text="Claim A", final_label="SUPPORTED", confidence=0.9),
                VerificationResult(claim_id="c2", claim_text="Claim B", final_label="SUPPORTED", confidence=0.9),
                VerificationResult(claim_id="c3", claim_text="Claim C", final_label="SUPPORTED", confidence=0.9)
            ]
            files = rg.generate_report(rq, v_res)
            with open(files["json_path"], "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["verification"]["supported"], 3)
            self.assertIn("3 supported", data["overall_synthesis"])


if __name__ == "__main__":
    unittest.main()
