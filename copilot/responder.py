"""[5] Final Response: template answer from evidence (ARCHITECTURE.md §6 [5]). The LLM draft is added in step 3."""

from copilot.schemas import CopilotResponse, DataResult, Decision, PolicyHit, SafetyCheck, TriageResult

NO_POLICY_LINE = "I cannot confirm the policy for this from the available documents."
MAX_POLICY_STEPS = 3

FOLLOW_UP_QUESTIONS = {
    "store_id": "Which store is this (e.g. ST002)?",
    "sku": "Which product or SKU is affected?",
    "amount_inr": "What is the refund amount in INR?",
    "receipt_present": "Does the customer have the receipt?",
}


# Top policy hit -> up to 3 bullet steps from its section ("- ..." lines), as next actions.
def policy_steps(hits: list[PolicyHit]) -> list[str]:
    if not hits:
        return []
    steps = []
    for line in hits[0].text.splitlines():
        if line.startswith("- ") and len(steps) < MAX_POLICY_STEPS:
            steps.append(line[2:].strip())
    return steps


# Triage missing fields -> one follow-up question, or None when nothing critical is missing.
def follow_up_for(missing_fields: list[str]) -> str | None:
    for field in missing_fields:
        if field in FOLLOW_UP_QUESTIONS:
            return FOLLOW_UP_QUESTIONS[field]
    return None


# Decision + hits + data -> template answer text (also the fallback when the LLM fails, step 3).
def template_answer(triage: TriageResult, hits: list[PolicyHit], data_results: list[DataResult],
                    decision: Decision) -> str:
    sentences = [f"Classified as {triage.category}, priority {decision.priority}."]
    if hits:
        sentences.append(f"Relevant policy: {hits[0].section}.")
    else:
        sentences.append(NO_POLICY_LINE)
    for result in data_results:
        sentences.extend(result.summary_lines[:2])
    if decision.escalation.required:
        sentences.append(f"Escalate to {decision.escalation.team}.")
    return " ".join(sentences)


# All stage outputs -> final CopilotResponse (template path; output gate added in step 3).
def compose(safety: SafetyCheck, triage: TriageResult, hits: list[PolicyHit],
            data_results: list[DataResult], decision: Decision) -> CopilotResponse:
    data_used = []
    for result in data_results:
        data_used.extend(name for name in result.data_used if name not in data_used)
    findings = {result.tool: result.findings for result in data_results}
    return CopilotResponse(
        trace_id="", mode=triage.mode, category=triage.category, priority=decision.priority,
        priority_rule_id=decision.rule_id, priority_reason=decision.priority_reason,
        confidence=triage.confidence, answer=template_answer(triage, hits, data_results, decision),
        next_actions=policy_steps(hits), follow_up_question=follow_up_for(triage.missing_fields),
        citations=decision.citations, data_used=data_used, data_findings=findings,
        escalation=decision.escalation, pii_masked=bool(safety.pii_found), safety_refusal=False,
        workflow_payload=decision.workflow_payload, limitations=None if hits else NO_POLICY_LINE,
    )
