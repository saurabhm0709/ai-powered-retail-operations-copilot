"""Deterministic rules: keyword classifier, escalation-matrix lookup, priority (ARCHITECTURE.md §6 [2], [4b])."""

import re

from copilot import config
from copilot.data_store import load_table
from copilot.schemas import Category, DataResult, Escalation, TriageResult

# Checked in order; first match wins. Order follows the SOP tie-break "choose the category linked to the
# immediate risk" (01_retail_operations_sop.md#service-categories): safety before pricing before POS, etc.
CATEGORY_KEYWORDS: list[tuple[Category, str]] = [
    ("Product Quality and Safety", r"injur|recall|unsafe|hazard|defect|seam|split|crack|warp|torn|malfunction"
                                   r"|quality complaint|broke|hurt|belt slip"),
    ("Pricing and Promotions", r"price|pricing|shelf label|promo|discount|offer|mrp|overcharg"),
    ("POS and Payments", r"register|\breg\b|card ?reader|card payment|card machine|\bpos\b|payment|upi|terminal|swipe"),
    ("Returns and Refunds", r"\breturn|refund|exchange"),
    ("Vendor and Supply Chain", r"short ship|shipment|received (less|fewer|only)|vendor|supplier|purchase order"
                                r"|earlier order|order of \d+|\bpo\b"),
    ("Inventory and Replenishment", r"out of stock|stock|not available|inventory|reorder|cycle count|replenish"
                                    r"|system\b.{0,40}\bshelf|shelf count|on.?hand"),
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
        for team in row["team"].split(" + "):  # matrix cells combine teams, e.g. "Security + POS Support"
            if team not in teams:
                teams.append(team)
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


SOP = "01_retail_operations_sop.md"
POS = "02_pos_pricing_returns_policy.md"
INV = "03_inventory_replenishment_policy.md"
VQ = "04_vendor_quality_policy.md"
RAI = "05_responsible_ai_policy.md"

MULTI_REGISTER_SCOPES = {"multiple_registers", "all_registers", "one_store"}
MULTI_PRICING_SCOPES = {"many_skus", "department", "many_customers", "multiple_stores"}

# Priority rules, ordered, first match wins (ARCHITECTURE.md §6 [4b], 01_retail_operations_sop.md#priority-rules).
# P1-SECURITY and P1-PAYMENT-DATA are decided by the input gate (controller.refusal), so they are not here.
# when(triage, findings, high_demand) -> bool. "trigger" is the escalation_matrix row (None = no row).
PRIORITY_RULES = [
    {"id": "P1-SAFETY", "priority": "P1",
     "when": lambda t, f, hd: t.facts.safety_signal != "none" or "safety_incident" in f,
     "citations": [f"{VQ}#recall-and-safety", f"{SOP}#priority-rules"],
     "trigger": "Contamination, injury, recall, or unsafe equipment",
     "reason": "Possible injury, recall or unsafe equipment is P1."},
    {"id": "P1-POS-OUTAGE", "priority": "P1",
     "when": lambda t, f, hd: t.category == "POS and Payments" and t.facts.scope in MULTI_REGISTER_SCOPES,
     "citations": [f"{POS}#pos-card-reader-runbook", f"{SOP}#priority-rules"],
     "trigger": "Multiple registers or store-wide payment outage",
     "reason": "Multiple registers or a store-wide payment outage is P1."},
    {"id": "P2-STOCKOUT", "priority": "P2",
     "when": lambda t, f, hd: "stockout_high_demand" in f or ("stockout" in f and t.facts.order_context != "none"),
     "citations": [f"{INV}#stock-outs-and-replenishment"],
     "trigger": "High-demand SKU stock-out or false availability",
     "reason": "Stock-out of a high-demand item or for a confirmed order is P2."},
    {"id": "P2-STOCKOUT", "priority": "P2",
     "when": lambda t, f, hd: "warehouse_depleted" in f,
     "citations": [f"{INV}#stock-outs-and-replenishment", f"{INV}#root-cause-hints"],
     "trigger": "Warehouse stock depleted for high-demand or academy-order item",
     "reason": "Warehouse stock is depleted for this item, which is P2."},
    {"id": "P2-SHORT-SHIPMENT", "priority": "P2",
     "when": lambda t, f, hd: ("short_shipment" in f and hd) or "consecutive_short_shipment" in f,
     "citations": [f"{VQ}#short-shipments"],
     "trigger": "Repeated or below-90-percent short shipment",
     "reason": "Short shipment below 90% on a high-demand item, or repeated short shipments, is P2."},
    {"id": "P2-PRICING-MULTI", "priority": "P2",
     "when": lambda t, f, hd: t.category == "Pricing and Promotions" and t.facts.scope in MULTI_PRICING_SCOPES,
     "citations": [f"{POS}#pricing-and-promotions"],
     "trigger": "Department or multi-customer pricing mismatch",
     "reason": "A pricing mismatch affecting many SKUs, customers or stores is P2."},
    {"id": "P2-QUALITY-REPEATED", "priority": "P2",
     "when": lambda t, f, hd: t.category == "Product Quality and Safety"
                              and (t.facts.repeated or t.facts.multi_store or "repeated_quality_complaints" in f),
     "citations": [f"{VQ}#product-quality-complaints"],
     "trigger": "Repeated quality complaints without immediate safety risk",
     "reason": "Repeated or multi-store quality complaints without immediate safety risk are P2."},
    {"id": "P2-REFUND-APPROVAL", "priority": "P2",
     "when": lambda t, f, hd: "refund_needs_approval" in f,
     "citations": [f"{POS}#returns-and-refunds", f"{RAI}#human-in-the-loop"],
     "trigger": "No-receipt or high-value refund threshold",
     "reason": "The refund is above an approval threshold, so it needs manager approval (P2)."},
    {"id": "P2-DISCREPANCY", "priority": "P2",
     # 03_inventory_replenishment_policy.md#inventory-discrepancy: P2 only when it causes false availability.
     "when": lambda t, f, hd: "false_availability" in f and hd,
     "citations": [f"{INV}#inventory-discrepancy"],
     "trigger": "High-demand SKU stock-out or false availability",
     "reason": "An inventory discrepancy on a high-demand item is P2."},
    {"id": "P3-POS-SINGLE", "priority": "P3",
     "when": lambda t, f, hd: t.category == "POS and Payments" and t.facts.scope == "one_register",
     "citations": [f"{POS}#pos-card-reader-runbook"],
     "trigger": "One register or one card reader failing",
     "reason": "One register or card reader failing while others work is P3."},
    # The next three also require mode "triage": a returns/pricing/service *analytics* question is P4, not P3.
    {"id": "P3-PRICING-SINGLE", "priority": "P3",
     "when": lambda t, f, hd: t.category == "Pricing and Promotions" and t.mode == "triage",
     "citations": [f"{POS}#pricing-and-promotions"],
     "trigger": "Single SKU shelf-vs-register mismatch",
     "reason": "A single-SKU shelf-vs-register mismatch is P3."},
    {"id": "P3-INVENTORY-ROUTINE", "priority": "P3",
     "when": lambda t, f, hd: "below_reorder" in f or "inventory_discrepancy" in f,
     "citations": [f"{INV}#inventory-discrepancy"],
     "trigger": "Routine reorder or cycle count",
     "reason": "Below reorder point or a routine count discrepancy is P3."},
    {"id": "P3-RETURN-STANDARD", "priority": "P3",
     "when": lambda t, f, hd: t.category == "Returns and Refunds" and t.mode == "triage",
     "citations": [f"{POS}#returns-and-refunds"],
     "trigger": None,
     "reason": "A standard return within policy limits is P3."},
    {"id": "P3-SERVICE", "priority": "P3",
     "when": lambda t, f, hd: t.category == "Customer Service" and t.mode == "triage",
     "citations": [f"{POS}#customer-recovery"],
     "trigger": "Routine service recovery",
     "reason": "Routine customer service recovery is P3."},
    {"id": "P4-ANALYTICS", "priority": "P4",
     "when": lambda t, f, hd: t.mode in ("analytics", "policy_question"),
     "citations": [f"{SOP}#analytics-rule"],
     "trigger": "General inquiry or analytics",
     "reason": "An analytics or general question with no active incident is P4."},
    {"id": "P3-DEFAULT", "priority": "P3",
     "when": lambda t, f, hd: True,
     "citations": [f"{SOP}#priority-rules"],
     "trigger": None,
     "reason": "An active issue with no specific rule defaults to P3."},
]


# Data results -> set of all finding codes.
def all_findings(data_results: list[DataResult]) -> set[str]:
    findings = set()
    for result in data_results:
        findings.update(result.findings)
    return findings


# Data results -> True if any tool reported the SKU as high-demand (facts["high_demand"]).
def is_high_demand(data_results: list[DataResult]) -> bool:
    return any(result.facts.get("high_demand") is True for result in data_results)


# Triage + data -> every rule whose condition holds, in table order (winner first).
def fired_rules(triage: TriageResult, data_results: list[DataResult]) -> list[dict]:
    findings = all_findings(data_results)
    high_demand = is_high_demand(data_results)
    fired = []
    for rule in PRIORITY_RULES:
        if rule["when"](triage, findings, high_demand):
            fired.append(rule)
    return fired


# Triage + data -> (priority, rule_id) of the first matching rule. Same function for preliminary and final.
def decide_priority(triage: TriageResult, data_results: list[DataResult]) -> tuple[str, str]:
    winner = fired_rules(triage, data_results)[0]
    return winner["priority"], winner["id"]


# Fired rules -> winner + other P1/P2 rules. Lower fired rules (e.g. P3-DEFAULT) add no escalation or citation.
def deciding_rules(fired: list[dict]) -> list[dict]:
    selected = [fired[0]]
    for rule in fired[1:]:
        if rule["priority"] in ("P1", "P2") and rule not in selected:
            selected.append(rule)
    return selected


# LLM confidence -> calibrated confidence (ARCHITECTURE.md §6 [2] step 4; 05_responsible_ai_policy.md#uncertainty).
def calibrate_confidence(llm_confidence: float, llm_category: str, keyword_cat: str,
                         missing_fields: list[str]) -> float:
    confidence = llm_confidence
    if llm_category != keyword_cat:
        confidence -= config.CATEGORY_DISAGREE_PENALTY
    # Small on purpose: a missing field is answered by the follow-up question, not by escalation.
    confidence -= min(config.MISSING_FIELD_PENALTY * len(missing_fields), config.MISSING_FIELD_PENALTY_MAX)
    return round(max(0.0, min(1.0, confidence)), 2)
