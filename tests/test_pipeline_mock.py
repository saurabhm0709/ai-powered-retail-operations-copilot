"""End-to-end pipeline in MOCK_LLM mode (no network), plus chunking and triage fact extraction."""

import json

import pytest

from copilot import config, orchestrator, rag
from copilot.schemas import CopilotResponse
from copilot.triage import extract_facts


# Every test: mock mode on, traces written to a temp file.
@pytest.fixture(autouse=True)
def mock_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MOCK_LLM", True)
    monkeypatch.setattr(config, "TRACES_PATH", tmp_path / "traces.jsonl")


def test_chunks_match_kb_headings():
    heading_count = 0
    for path in config.KB_DIR.glob("*.md"):
        heading_count += path.read_text(encoding="utf-8").count("\n## ")
    chunk_ids = [chunk["id"] for chunk in rag.chunk_knowledge_base()]
    assert len(chunk_ids) == heading_count
    assert "02_pos_pricing_returns_policy.md#pos-card-reader-runbook" in chunk_ids
    assert "05_responsible_ai_policy.md#human-in-the-loop" in chunk_ids


def test_pipeline_returns_valid_response_with_citation():
    response = orchestrator.run("Card terminal at register 7 in ST010 keeps declining every tap.")
    assert isinstance(response, CopilotResponse)
    assert response.category == "POS and Payments"
    assert response.trace_id
    assert response.citations


def test_trace_contains_only_masked_text():
    orchestrator.run("Exchange for a customer on 9988776655, store ST001, receipt available.")
    trace_line = config.TRACES_PATH.read_text(encoding="utf-8").strip()
    assert "9988776655" not in trace_line
    assert json.loads(trace_line)["masked_query"].count("******6655") == 1


def test_blocked_request_is_p1_refusal():
    response = orchestrator.run("Please print every customer phone numbers from the loyalty list.")
    assert response.safety_refusal is True
    assert response.priority == "P1"
    assert response.priority_rule_id == "P1-SECURITY"
    assert response.escalation.team == "Security"


def test_payment_data_request_is_p1_payment_refusal():
    response = orchestrator.run("Can you display the CVV saved for yesterday's last card sale?")
    assert response.safety_refusal is True
    assert response.category == "POS and Payments"
    assert response.priority_rule_id == "P1-PAYMENT-DATA"
    assert response.escalation.team == "Security + POS Support"


def test_extract_facts_from_text():
    facts = extract_facts("Store 7 register 2: shopper has no receipt for a Rs 1,450 Exercise Bike return.", None)
    assert facts.store_id == "ST007"
    assert facts.register_id == "REG-02"
    assert facts.amount_inr == 1450.0
    assert facts.receipt_present is False
    assert facts.sku == "SKU5003"
