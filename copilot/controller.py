"""[4b] Safety & Workflow Controller — decision step and blocked-request refusal (ARCHITECTURE.md §6 [4a], [4b])."""

from copilot import config, rules, workflow
from copilot.schemas import CopilotResponse, DataResult, Decision, PolicyHit, SafetyCheck, TriageResult

INJECTION_CITATION = "05_responsible_ai_policy.md#prompt-injection"
PAYMENT_CITATION = "02_pos_pricing_returns_policy.md#payment-security"
SECURITY_TRIGGER = "Restricted data, prompt injection, or suspicious request"
PAYMENT_TRIGGER = "Payment data or credential request"

# Data tool -> policy sections it applies ("finding citations", ARCHITECTURE.md §1 [4b]). Tools that analyse
# CSV data to answer a question also cite the SOP analytics rule (01_retail_operations_sop.md#analytics-rule).
ANALYTICS_CITATION = f"{rules.SOP}#analytics-rule"
TOOL_CITATIONS = {
    "stock_status": [f"{rules.INV}#stock-outs-and-replenishment"],
    "receipt_history": [f"{rules.VQ}#short-shipments"],
    "warehouse_status": [f"{rules.INV}#root-cause-hints"],
    "vendor_performance": [f"{rules.VQ}#vendor-performance"],
    "stockout_root_cause": [f"{rules.INV}#root-cause-hints", ANALYTICS_CITATION],
    "quality_complaints": [f"{rules.VQ}#product-quality-complaints"],
    "vendor_quality_ranking": [f"{rules.VQ}#vendor-performance", ANALYTICS_CITATION],
    "case_query": [ANALYTICS_CITATION],
    "short_shipment_summary": [ANALYTICS_CITATION],
    "returns_summary": [ANALYTICS_CITATION],
    "refund_check": [f"{rules.POS}#returns-and-refunds"],
    "price_lookup": [f"{rules.POS}#pricing-and-promotions"],
}


# Blocked SafetyCheck -> (category, rule_id, matrix trigger, citations) per ARCHITECTURE.md §6 [4a].
def refusal_route(safety: SafetyCheck) -> tuple[str, str, str, list[str]]:
    if safety.threat_type == "payment_data_request":
        return "POS and Payments", "P1-PAYMENT-DATA", PAYMENT_TRIGGER, [PAYMENT_CITATION]
    return "Other or Security", "P1-SECURITY", SECURITY_TRIGGER, [INJECTION_CITATION, PAYMENT_CITATION]


# Blocked SafetyCheck -> refusal CopilotResponse: P1, security escalation, nothing revealed, no LLM calls.
def refusal(safety: SafetyCheck) -> CopilotResponse:
    category, rule_id, trigger, citations = refusal_route(safety)
    escalation = rules.build_escalation([trigger], confidence=1.0, priority="P1", blocked=True)
    return CopilotResponse(
        trace_id="", mode="triage", category=category, priority="P1", priority_rule_id=rule_id,
        priority_reason=safety.block_reason or "Request blocked by the input safety gate.",
        confidence=1.0,
        answer="I can't help with that request. Card data, credentials and restricted business or customer "
               "data are never shared through the copilot. It has been flagged for Security review.",
        next_actions=["Do not share payment or customer data in chat.",
                      f"Raise legitimate access needs with {escalation.team} via {escalation.channel}."],
        follow_up_question=None, citations=citations, data_used=[], data_findings={},
        escalation=escalation, pii_masked=bool(safety.pii_found), safety_refusal=True,
        workflow_payload=workflow.security_payload(safety, category, rule_id, escalation), limitations=None,
    )


# Retrieved hits -> citation ids in rank order, deduplicated.
def hit_citations(hits: list[PolicyHit]) -> list[str]:
    citations = []
    for hit in hits:
        if hit.citation not in citations:
            citations.append(hit.citation)
    return citations


# Deciding rules + data + hits -> ordered, deduplicated citations: rule citations first, then data-tool
# citations, then retrieved.
def ordered_citations(selected_rules: list[dict], data_results: list[DataResult],
                      hits: list[PolicyHit]) -> list[str]:
    citations = []
    for rule in selected_rules:
        for citation in rule["citations"]:
            if citation not in citations:
                citations.append(citation)
    for result in data_results:
        for citation in TOOL_CITATIONS.get(result.tool, []):
            if citation not in citations:
                citations.append(citation)
    for citation in hit_citations(hits):
        if citation not in citations:
            citations.append(citation)
    return citations


# Safety + triage + hits + data -> Decision: final priority by rules, escalation from the matrix rows of
# the deciding rules, citations from rules + data tools + retrieval, and the mock workflow payload.
def decide(safety: SafetyCheck, triage: TriageResult, hits: list[PolicyHit],
           data_results: list[DataResult]) -> Decision:
    selected = rules.deciding_rules(rules.fired_rules(triage, data_results))
    winner = selected[0]
    triggers = [rule["trigger"] for rule in selected if rule["trigger"]]
    escalation = rules.build_escalation(triggers, triage.confidence, winner["priority"], blocked=False)
    allowed = ordered_citations(selected, data_results, hits)
    payload = workflow.build_payload(safety, triage, winner["priority"], winner["id"], winner["reason"],
                                     escalation, data_results)
    return Decision(
        priority=winner["priority"], rule_id=winner["id"], priority_reason=winner["reason"],
        fired_rule_ids=[rule["id"] for rule in selected], allowed_citations=allowed,
        citations=allowed[:config.MAX_CITATIONS], escalation=escalation, workflow_payload=payload,
    )
