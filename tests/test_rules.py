"""Tests for rules.py: keyword classifier, priority table, confidence calibration, escalation."""

from copilot import rules
from copilot.schemas import DataResult, Facts, TriageResult
from copilot.triage import extract_pos_severity


# Minimal TriageResult for rule tests; override any field with keyword arguments.
def make_triage(category: str = "Other or Security", mode: str = "triage", **fact_values) -> TriageResult:
    facts = Facts(store_id=None, product_mention=None, sku=None, register_id=None,
                  amount_inr=None, receipt_present=None, time_window_hours=None, **fact_values)
    return TriageResult(mode=mode, category=category, llm_confidence=0.7, confidence=0.7, facts=facts,
                        needs_policy=True, needs_data=False, analytics_intent=None,
                        preliminary_priority="P3", preliminary_rule_id="P3-DEFAULT",
                        missing_fields=[], short_reasoning="")


# One DataResult carrying the given finding codes and high_demand flag.
def make_data(findings: list[str], high_demand: bool = False) -> list[DataResult]:
    return [DataResult(tool="test", facts={"high_demand": high_demand}, findings=findings,
                       summary_lines=[], data_used=[])]


def test_keyword_category_prefers_immediate_risk():
    assert rules.keyword_category("Seam on a football split open, customer wants a refund") == \
        "Product Quality and Safety"
    assert rules.keyword_category("Label on the shelf shows a lower price than the register") == \
        "Pricing and Promotions"
    assert rules.keyword_category("UPI payments failing on the counter terminal") == "POS and Payments"


def test_multi_register_outage_is_p1():
    triage = make_triage("POS and Payments", scope="all_registers", pos_severity="outage")
    assert rules.decide_priority(triage, []) == ("P1", "P1-POS-OUTAGE")


def test_multi_register_unknown_severity_fails_safe_to_p1():
    triage = make_triage("POS and Payments", scope="multiple_registers")   # pos_severity defaults to "unknown"
    assert rules.decide_priority(triage, []) == ("P1", "P1-POS-OUTAGE")


def test_multi_register_degraded_is_p2_with_matrix_team():
    triage = make_triage("POS and Payments", scope="multiple_registers", pos_severity="degraded")
    assert rules.decide_priority(triage, []) == ("P2", "P2-POS-DEGRADED")
    winner = rules.fired_rules(triage, [])[0]
    escalation = rules.build_escalation([winner["trigger"]], 0.7, "P2", blocked=False)
    assert escalation.team == "Store Manager + POS Support"


def test_pos_severity_keywords():
    assert extract_pos_severity("Registers 3 and 4 cannot take card payments") == "outage"
    assert extract_pos_severity("Card payments on several registers are slow and timing out") == "degraded"
    assert extract_pos_severity("Tills are slow and now register 2 is down") == "outage"   # outage wins
    assert extract_pos_severity("Question about register float counts") == "unknown"


def test_single_register_is_p3():
    triage = make_triage("POS and Payments", scope="one_register")
    assert rules.decide_priority(triage, []) == ("P3", "P3-POS-SINGLE")


def test_safety_signal_beats_everything():
    triage = make_triage("POS and Payments", scope="all_registers", safety_signal="injury")
    assert rules.decide_priority(triage, [])[1] == "P1-SAFETY"


def test_stockout_needs_high_demand_or_order():
    plain = make_triage("Inventory and Replenishment")
    assert rules.decide_priority(plain, make_data(["stockout"]))[0] == "P3"
    with_order = make_triage("Inventory and Replenishment", order_context="school")
    assert rules.decide_priority(with_order, make_data(["stockout"])) == ("P2", "P2-STOCKOUT")


def test_short_shipment_needs_high_demand():
    triage = make_triage("Vendor and Supply Chain")
    assert rules.decide_priority(triage, make_data(["short_shipment"]))[0] == "P3"
    assert rules.decide_priority(triage, make_data(["short_shipment"], high_demand=True))[1] == "P2-SHORT-SHIPMENT"


def test_returns_analytics_is_p4_not_p3():
    triage = make_triage("Returns and Refunds", mode="analytics")
    assert rules.decide_priority(triage, []) == ("P4", "P4-ANALYTICS")


def test_deciding_rules_keep_secondary_p2_only():
    triage = make_triage("Inventory and Replenishment")
    fired = rules.fired_rules(triage, make_data(["stockout_high_demand", "consecutive_short_shipment", "below_reorder"]))
    selected_ids = [rule["id"] for rule in rules.deciding_rules(fired)]
    assert selected_ids == ["P2-STOCKOUT", "P2-SHORT-SHIPMENT"]


def test_calibrate_confidence():
    assert rules.calibrate_confidence(0.9, "POS and Payments", "POS and Payments", []) == 0.9
    assert rules.calibrate_confidence(0.9, "POS and Payments", "Customer Service", []) == 0.7
    assert rules.calibrate_confidence(0.7, "A", "A", ["store_id"]) == 0.65
    assert rules.calibrate_confidence(0.7, "A", "A", ["store_id", "sku", "amount_inr"]) == 0.6  # max -0.10
    # Missing fields alone never push the mock confidence (0.7) below the 0.60 escalation threshold.
    assert rules.calibrate_confidence(0.7, "A", "A", ["store_id", "sku"]) >= 0.60


def test_escalation_merges_teams_and_takes_min_sla():
    triggers = ["High-demand SKU stock-out or false availability", "Repeated or below-90-percent short shipment"]
    escalation = rules.build_escalation(triggers, confidence=0.8, priority="P2", blocked=False)
    assert escalation.teams == ["Store Manager", "Inventory Lead", "Supply Chain"]
    assert escalation.sla_minutes == 60
    assert escalation.required is True  # short-shipment row needs human approval


def test_low_confidence_without_row_goes_to_store_manager():
    escalation = rules.build_escalation([], confidence=0.5, priority="P3", blocked=False)
    assert escalation.required is True
    assert escalation.team == "Store Manager"
    assert escalation.reason == "low confidence"
