"""Deterministic rules: keyword classifier, escalation-matrix lookup, priority (ARCHITECTURE.md §6 [2], [4b])."""

import re

from copilot import config
from copilot.data_store import load_table
from copilot.schemas import Category, Escalation, TriageResult

# Checked in order; first match wins. Order follows the SOP tie-break "choose the category linked to the
# immediate risk" (01_retail_operations_sop.md#service-categories): safety before pricing before POS, etc.
CATEGORY_KEYWORDS: list[tuple[Category, str]] = [
    ("Product Quality and Safety", r"injur|recall|unsafe|hazard|defect|seam|split|crack|warp|torn|malfunction"
                                   r"|quality complaint|broke"),
    ("Pricing and Promotions", r"price|pricing|shelf label|promo|discount|offer|mrp|overcharg"),
    ("POS and Payments", r"register|card reader|card payment|card machine|\bpos\b|payment|upi|terminal|swipe"),
    ("Returns and Refunds", r"\breturn|refund|exchange"),
    ("Vendor and Supply Chain", r"short ship|shipment|received (less|fewer|only)|vendor|supplier|purchase order"),
    ("Inventory and Replenishment", r"out of stock|stock|not available|inventory|reorder|cycle count|replenish"),
    ("Customer Service", r"complain|rude|waiting|service|loyalty|upset"),
]


# Masked text -> category by keyword (MOCK_LLM classifier and the disagreement check for confidence).
def keyword_category(masked_text: str) -> Category:
    lowered = masked_text.lower()
    for category, pattern in CATEGORY_KEYWORDS:
        if re.search(pattern, lowered):
            return category
    return "Other or Security"


# Matrix trigger text -> its escalation_matrix row as a dict.
def matrix_row(trigger: str) -> dict:
    matrix = load_table("escalation_matrix")
    return matrix[matrix["trigger"] == trigger].iloc[0].to_dict()


# Matrix triggers + confidence + blocked flag -> Escalation (01_retail_operations_sop.md#escalation-policy).
def build_escalation(triggers: list[str], confidence: float, priority: str, blocked: bool) -> Escalation:
    rows = [matrix_row(trigger) for trigger in triggers]
    teams = []
    for row in rows:
        if row["team"] not in teams:
            teams.append(row["team"])
    approval = any(row["human_approval_required"] for row in rows)
    low_confidence = confidence < config.CONFIDENCE_ESCALATION
    if low_confidence and not teams:
        teams = ["Store Manager"]
    required = priority == "P1" or low_confidence or blocked or approval
    return Escalation(
        required=required, teams=teams, team=" + ".join(teams),
        channel=rows[0]["channel"] if rows else None,
        sla_minutes=min(int(row["sla_minutes"]) for row in rows) if rows else None,
        human_approval_required=approval,
        reason="low confidence" if low_confidence and not rows else "; ".join(triggers),
    )


# Triage + data results -> (priority, rule_id). PLACEHOLDER until the full ordered rule table (step 2).
def decide_priority(triage: TriageResult, data_results: list) -> tuple[str, str]:
    if triage.mode in ("analytics", "policy_question"):
        return "P4", "P4-ANALYTICS"
    return "P3", "P3-DEFAULT"
