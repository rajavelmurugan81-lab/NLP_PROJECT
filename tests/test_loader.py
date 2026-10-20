"""Unit tests for src/data/loader.py."""

import os
import json
import pytest
from src.data.loader import (
    SciFactLoader,
    CoVERTLoader,
    SciClaimHuntLoader,
    get_loader
)


def test_scifact_loader_execution(tmp_path):
    # Setup mock raw SciFact folder
    raw_dir = tmp_path / "scifact"
    raw_dir.mkdir()
    
    claims_file = raw_dir / "claims_train.jsonl"
    corpus_file = raw_dir / "corpus.jsonl"

    corpus_file.write_text(json.dumps({
        "doc_id": 101,
        "title": "Gene Editing",
        "abstract": ["Sentence zero.", "Sentence one."],
        "structured": False
    }) + "\n", encoding="utf-8")

    claims_file.write_text(json.dumps({
        "id": 1,
        "claim": "Gene editing works.",
        "evidence": {
            "101": [{"sentences": [0, 1], "label": "SUPPORT"}]
        },
        "cited_doc_ids": [101]
    }) + "\n", encoding="utf-8")

    loader = SciFactLoader(raw_dir=str(raw_dir))
    records = loader.load()

    assert len(records) == 1
    rec = records[0]
    assert rec.claim == "Gene editing works."
    assert rec.label == "SUPPORT"
    assert rec.evidence_id == "101"
    assert "Sentence zero. Sentence one." in rec.evidence_text
    assert rec.evidence_title == "Gene Editing"


def test_covert_loader_execution(tmp_path):
    raw_dir = tmp_path / "covert"
    raw_dir.mkdir()
    covert_file = raw_dir / "covert.json"

    data = [
        {
            "claim_id": "cov_01",
            "claim": "COVID vaccine claim",
            "evidence": "Medical trial evidence",
            "verdict": "SUPPORT",
            "split": "train"
        }
    ]
    covert_file.write_text(json.dumps(data), encoding="utf-8")

    loader = CoVERTLoader(raw_dir=str(raw_dir))
    records = loader.load()

    assert len(records) == 1
    assert records[0].claim_id == "cov_01"
    assert records[0].label == "SUPPORT"
    assert records[0].evidence_text == "Medical trial evidence"


def test_sciclaimhunt_loader_execution(tmp_path):
    raw_dir = tmp_path / "sciclaimhunt"
    raw_dir.mkdir()
    sch_file = raw_dir / "sciclaimhunt_train.jsonl"

    item = {
        "claim_id": "sch_01",
        "claim": "Pruning reduces model size",
        "section_name": "Results",
        "evidence_spans": ["Pruning removed 80% parameters"],
        "label": "SUPPORTED",
        "split": "train"
    }
    sch_file.write_text(json.dumps(item) + "\n", encoding="utf-8")

    loader = SciClaimHuntLoader(raw_dir=str(raw_dir))
    records = loader.load()

    assert len(records) == 1
    assert records[0].claim_id == "sch_01"
    assert records[0].label == "SUPPORT"
    assert "80% parameters" in records[0].evidence_text


def test_get_loader_factory():
    scifact_l = get_loader("scifact")
    assert isinstance(scifact_l, SciFactLoader)

    covert_l = get_loader("covert")
    assert isinstance(covert_l, CoVERTLoader)

    sch_l = get_loader("sciclaimhunt")
    assert isinstance(sch_l, SciClaimHuntLoader)

    with pytest.raises(ValueError):
        get_loader("non_existent_dataset")


def test_malformed_input_handling(tmp_path):
    raw_dir = tmp_path / "covert_bad"
    raw_dir.mkdir()
    covert_file = raw_dir / "covert.json"
    covert_file.write_text("[{malformed_json: true]", encoding="utf-8")

    loader = CoVERTLoader(raw_dir=str(raw_dir))
    with pytest.raises(ValueError):
        loader.load()
