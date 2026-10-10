"""[4b] Safety & Workflow Controller — decision step and blocked-request refusal (ARCHITECTURE.md §6 [4a], [4b])."""

from copilot import config, rules
from copilot.schemas import CopilotResponse, DataResult, Decision, PolicyHit, SafetyCheck, TriageResult

INJECTION_CITATION = "05_responsible_ai_policy.md#prompt-injection"
PAYMENT_CITATION = "02_pos_pricing_returns_policy.md#payment-security"
SECURITY_TRIGGER = "Restricted data, prompt injection, or suspicious request"
PAYMENT_TRIGGER = "Payment data or credential request"


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
        workflow_payload=None, limitations=None,
    )


# Retrieved hits -> citation ids in rank order, deduplicated.
def hit_citations(hits: list[PolicyHit]) -> list[str]:
    citations = []
    for hit in hits:
        if hit.citation not in citations:
            citations.append(hit.citation)
    return citations


# Deciding rules + hits -> ordered, deduplicated citations: rule citations first, then retrieved.
def ordered_citations(selected_rules: list[dict], hits: list[PolicyHit]) -> list[str]:
    citations = []
    for rule in selected_rules:
        for citation in rule["citations"]:
            if citation not in citations:
                citations.append(citation)
    for citation in hit_citations(hits):
        if citation not in citations:
            citations.append(citation)
    return citations


# Safety + triage + hits + data -> Decision: final priority by rules, escalation from the matrix rows of
# the deciding rules, citations from rules + retrieval. Workflow payload is added in step 3.
def decide(safety: SafetyCheck, triage: TriageResult, hits: list[PolicyHit],
           data_results: list[DataResult]) -> Decision:
    selected = rules.deciding_rules(rules.fired_rules(triage, data_results))
    winner = selected[0]
    triggers = [rule["trigger"] for rule in selected if rule["trigger"]]
    escalation = rules.build_escalation(triggers, triage.confidence, winner["priority"], blocked=False)
    allowed = ordered_citations(selected, hits)
    return Decision(
        priority=winner["priority"], rule_id=winner["id"], priority_reason=winner["reason"],
        fired_rule_ids=[rule["id"] for rule in selected], allowed_citations=allowed,
        citations=allowed[:config.MAX_CITATIONS], escalation=escalation, workflow_payload=None,
    )
