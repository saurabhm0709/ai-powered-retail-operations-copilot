"""[5] Final Response: LLM draft from an evidence pack, checked by the output gate, with a template fallback
(ARCHITECTURE.md §6 [5]). The LLM only writes; priority, escalation and citations were decided by code."""

import json

from copilot import config, data_agent, llm, rag, safety
from copilot.schemas import (CopilotResponse, DataResult, Decision, PolicyHit, ResponseLLMOutput, SafetyCheck,
                             TriageResult)

NO_POLICY_LINE = "I cannot confirm the policy for this from the available documents."
UNCONFIRMED_POLICY_LINE = "I can't confirm which policy applies from this question alone."
# Asked when confidence is low but no specific field is missing (05_responsible_ai_policy.md#uncertainty).
GENERIC_FOLLOW_UP = ("Can you tell me what is happening, where it is happening (store id, register or area), "
                     "and which product is involved?")
# P1 template fallback opens with immediate safe guidance before anything else (ARCHITECTURE.md §6 [5]).
P1_GUIDANCE = "Immediate action: keep people safe, stop using the affected item or process, and escalate now."
MAX_POLICY_STEPS = 3
MAX_POLICY_SECTIONS = 6       # sections sent to the LLM (allowed citations, rule ones first)
SECTION_CHARS = 900           # policy sections are short; this trims only the longest ones

FOLLOW_UP_QUESTIONS = {
    "store_id": "Which store is this (e.g. ST002)?",
    "sku": "Which product or SKU is affected?",
    "amount_inr": "What is the refund amount in INR?",
    "receipt_present": "Does the customer have the receipt?",
}

RESPONSE_SYSTEM_PROMPT = """You write the reply of an internal retail operations copilot for Punjab Sports Mart staff.
You receive a JSON evidence pack. Priority, escalation and citations were already decided by code.

RULES:
- Use only the evidence pack. Do not change the priority, the escalation team or the SLA.
- Every number (stock, quantities, prices, rates, counts, dates) must come from data[].facts or data[].summary_lines.
  Never calculate new numbers.
- Cite only ids from allowed_citations, in "citations". Never invent a policy, contact, approval or availability.
- P1: give immediate safe guidance first.
- Never say a refund, return, claim, transfer or quarantine is approved, processed or done, and never say
  equipment is safe. Say it requires manager or human approval / review.
- Neutral language: "likely linked to", "requires review". Never blame staff, customers or vendors.
- Do not ask for or repeat card numbers, CVV, OTP, PINs or phone numbers; masked values stay masked.
- answer: at most 5 sentences. next_actions: at most 5 short imperative steps.
- follow_up_question: one question only when missing_fields is not empty (offer sku_candidates if given), else null.
"""


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


# Code decides whether a follow-up is asked: a critical field is missing, or confidence is below the
# escalation threshold (then a generic "what / where / which" question unless the draft has one).
def choose_follow_up(draft_question: str | None, triage: TriageResult) -> str | None:
    if triage.missing_fields:
        return draft_question or follow_up_for(triage.missing_fields)
    if triage.confidence < config.CONFIDENCE_ESCALATION:
        return draft_question or GENERIC_FOLLOW_UP
    return None


# Decision + hits + data -> template answer text (mock mode and the fallback when the LLM fails).
def template_answer(triage: TriageResult, hits: list[PolicyHit], data_results: list[DataResult],
                    decision: Decision) -> str:
    sentences = [P1_GUIDANCE] if decision.priority == "P1" else []
    sentences.append(f"Classified as {triage.category}, priority {decision.priority}.")
    # When the policy is unconfirmed, compose() adds UNCONFIRMED_POLICY_LINE instead.
    if decision.policy_confirmed:
        sentences.append(f"Relevant policy: {hits[0].section}." if hits else NO_POLICY_LINE)
    for result in data_results:
        sentences.extend(result.summary_lines[:2])
    if decision.escalation.required:
        sentences.append(f"Escalate to {decision.escalation.team}.")
    return " ".join(sentences)


# Allowed citations -> [{citation, text}] policy sections for the evidence pack (rule citations first).
def policy_sections(allowed_citations: list[str]) -> list[dict]:
    sections = []
    for citation in allowed_citations[:MAX_POLICY_SECTIONS]:
        sections.append({"citation": citation, "text": rag.get_section(citation)[:SECTION_CHARS]})
    return sections


# Stage outputs -> JSON evidence pack. Masked text only; numbers only from Data Agent facts.
def evidence_pack(safety_check: SafetyCheck, triage: TriageResult, data_results: list[DataResult],
                  decision: Decision) -> str:
    candidates = [f"{sku} {data_agent.product_name(sku)}" for sku in triage.facts.sku_candidates]
    escalation = decision.escalation.model_dump(include={"required", "team", "channel", "sla_minutes",
                                                         "human_approval_required"})
    pack = {
        "question": safety_check.masked_text, "mode": triage.mode, "category": triage.category,
        "decision": {"priority": decision.priority, "rule_id": decision.rule_id,
                     "priority_reason": decision.priority_reason, "escalation": escalation},
        "missing_fields": triage.missing_fields, "sku_candidates": candidates,
        "data": [result.model_dump(include={"tool", "findings", "summary_lines", "facts"})
                 for result in data_results],
        "policy_sections": policy_sections(decision.allowed_citations),
        "allowed_citations": decision.allowed_citations,
    }
    return json.dumps(pack, ensure_ascii=False, default=str)


# Code-decided citations first, then any extra allowed ones the LLM used, capped (ARCHITECTURE.md §6 [4b]).
def final_citations(decision: Decision, llm_citations: list[str]) -> list[str]:
    citations = list(decision.citations)
    for citation in llm_citations:
        if citation not in citations:
            citations.append(citation)
    return citations[:config.MAX_CITATIONS]


# Stage outputs -> checked LLM draft, or None (mock mode, LLM failure or output gate rejection).
def llm_draft(safety_check: SafetyCheck, triage: TriageResult, data_results: list[DataResult],
              decision: Decision) -> ResponseLLMOutput | None:
    draft, _usage = llm.structured(RESPONSE_SYSTEM_PROMPT, evidence_pack(safety_check, triage, data_results, decision),
                                   ResponseLLMOutput, config.RESPONSE_MAX_TOKENS)
    if draft is None:
        return None
    return safety.check_output(draft, decision.allowed_citations)


# Template path -> the same fields as an LLM draft, so compose() treats both paths alike.
def template_draft(triage: TriageResult, hits: list[PolicyHit], data_results: list[DataResult],
                   decision: Decision) -> ResponseLLMOutput:
    return ResponseLLMOutput(answer=template_answer(triage, hits, data_results, decision),
                             next_actions=policy_steps(hits) if decision.policy_confirmed else [],
                             follow_up_question=None,
                             citations=[])


# All stage outputs -> final CopilotResponse (LLM draft through the output gate, else the template).
def compose(safety_check: SafetyCheck, triage: TriageResult, hits: list[PolicyHit],
            data_results: list[DataResult], decision: Decision) -> CopilotResponse:
    draft = llm_draft(safety_check, triage, data_results, decision)
    if draft is None:
        draft = template_draft(triage, hits, data_results, decision)
    answer, limitations = draft.answer, None if hits else NO_POLICY_LINE
    if not decision.policy_confirmed:
        answer, limitations = f"{answer} {UNCONFIRMED_POLICY_LINE}", UNCONFIRMED_POLICY_LINE
    # P1: code guarantees safe guidance comes first, whatever the LLM wrote (template already starts with it).
    if decision.priority == "P1" and not answer.startswith(P1_GUIDANCE):
        answer = f"{P1_GUIDANCE} {answer}"
    data_used = []
    for result in data_results:
        data_used.extend(name for name in result.data_used if name not in data_used)
    return CopilotResponse(
        trace_id="", mode=triage.mode, category=triage.category, priority=decision.priority,
        priority_rule_id=decision.rule_id, priority_reason=decision.priority_reason,
        confidence=triage.confidence, answer=answer, next_actions=draft.next_actions,
        follow_up_question=choose_follow_up(draft.follow_up_question, triage),
        citations=final_citations(decision, draft.citations),
        data_used=data_used, data_findings={result.tool: result.findings for result in data_results},
        escalation=decision.escalation, pii_masked=bool(safety_check.pii_found), safety_refusal=False,
        workflow_payload=decision.workflow_payload, limitations=limitations,
    )
