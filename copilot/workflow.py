"""Mock workflow payloads: ticket / escalation / vendor-claim draft / security incident (ARCHITECTURE.md §6 [4b]).

Fields follow the SOP escalation summary (01_retail_operations_sop.md#escalation-policy). Nothing is sent
anywhere: payloads are appended to outputs/tickets.jsonl, and approval-bound ones stay PENDING_HUMAN_APPROVAL.
"""

import json
import uuid
from datetime import datetime

from copilot import config
from copilot.schemas import DataResult, Escalation, SafetyCheck, TriageResult

MAX_SUMMARY_CHARS = 300
MAX_SUPPORTING_LINES = 4


# Findings + escalation -> payload type. A short shipment needs a vendor claim review via Supply Chain
# (03_inventory_replenishment_policy.md#transfer-and-reorder-recommendation); the claim is only a draft.
def payload_type(findings: set[str], escalation: Escalation) -> str:
    if "short_shipment" in findings or "consecutive_short_shipment" in findings:
        return "vendor_claim_draft"
    if escalation.required:
        return "escalation"
    return "ticket"


# Shared fields of every payload. status waits for a human whenever approval is required.
def base_payload(payload_kind: str, category: str, priority: str, rule_id: str, reason: str,
                 escalation: Escalation, masked_text: str) -> dict:
    return {
        "ticket_id": f"TKT-{uuid.uuid4().hex[:8].upper()}",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "type": payload_kind, "category": category, "priority": priority,
        "policy_reason": f"{rule_id}: {reason}",
        "recommended_team": escalation.team or None, "channel": escalation.channel,
        "sla_minutes": escalation.sla_minutes,
        "status": "PENDING_HUMAN_APPROVAL" if escalation.human_approval_required else "OPEN",
        "summary": masked_text[:MAX_SUMMARY_CHARS],  # masked text only (AGENTS.md rule 2)
    }


# Triage + decision parts + data -> ticket / escalation / vendor-claim-draft payload.
def build_payload(safety_check: SafetyCheck, triage: TriageResult, priority: str, rule_id: str, reason: str,
                  escalation: Escalation, data_results: list[DataResult]) -> dict:
    findings = set()
    supporting = []
    for result in data_results:
        findings.update(result.findings)
        supporting.extend(result.summary_lines)
    payload = base_payload(payload_type(findings, escalation), triage.category, priority, rule_id, reason,
                           escalation, safety_check.masked_text)
    payload.update(store_id=triage.facts.store_id, sku=triage.facts.sku, impact_scope=triage.facts.scope,
                   supporting_data=supporting[:MAX_SUPPORTING_LINES])
    return payload


# Blocked request -> security incident payload. The request text is stored masked; nothing else is attached.
def security_payload(safety_check: SafetyCheck, category: str, rule_id: str, escalation: Escalation) -> dict:
    payload = base_payload("security_incident", category, "P1", rule_id, safety_check.block_reason or "",
                           escalation, safety_check.masked_text)
    payload.update(store_id=None, sku=None, impact_scope="unknown", supporting_data=[safety_check.threat_type])
    return payload


# Payload -> one JSON line appended to outputs/tickets.jsonl (the mock workflow system).
def append_ticket(payload: dict) -> None:
    config.TICKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TICKETS_PATH, "a", encoding="utf-8") as tickets_file:
        tickets_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
