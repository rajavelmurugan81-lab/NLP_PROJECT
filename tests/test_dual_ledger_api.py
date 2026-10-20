"""Tests for Dual-Ledger API Endpoints (/api/ledger/benchmark and /api/ledger/runtime)."""

import pytest
from fastapi.testclient import TestClient
from src.ui.server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_get_benchmark_ledger(client):
    response = client.get("/api/ledger/benchmark?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_benchmark_records"] >= 160
    assert len(data["records"]) <= 10
    assert "statistics" in data
    # Verify benchmark fields
    sample = data["records"][0]
    assert "claim_id" in sample
    assert "failure_type" in sample


def test_get_runtime_audit_ledger(client):
    response = client.get("/api/ledger/runtime?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total_runtime_audits" in data
    assert isinstance(data["records"], list)


def test_get_trusted_memory_ledger(client):
    response = client.get("/api/ledger/trusted")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "total_trusted_claims" in data
    assert data["total_trusted_claims"] >= 200
    assert "claims" in data
    assert len(data["claims"]) == data["total_trusted_claims"]
    
    sample = data["claims"][0]
    assert "claim_id" in sample
    assert "claim_text" in sample
    assert "verdict" in sample
    assert "confidence" in sample
    assert "evidence_count" in sample
    assert "question" in sample
    assert "timestamp" in sample

