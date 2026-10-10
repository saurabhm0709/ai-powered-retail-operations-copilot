"""Tests for data_store.py and the Data Agent tools against the ARCHITECTURE.md §9 data facts.

Uses the provided read-only CSVs. Tools are called directly with store/SKU ids, never with test prompts.
"""

import pandas as pd

from copilot import config, data_agent, safety, triage
from copilot.data_store import before_anchor, load_table, within_window


def test_timestamps_parsed_and_rows_after_anchor_excluded():
    cases = load_table("support_cases")
    assert pd.api.types.is_datetime64_any_dtype(cases["created_ts"])
    assert (cases["created_ts"] > config.ANCHOR).any()  # the data really has future rows
    kept = before_anchor(cases, "created_ts")
    assert kept["created_ts"].max() <= config.ANCHOR
    assert len(kept) < len(cases)


def test_yes_no_columns_are_booleans():
    products = load_table("product_master")
    assert products["high_demand"].dtype == bool
    assert products["high_demand"].any()


def test_blank_text_cells_are_empty_strings():
    cases = load_table("support_cases")
    assert not cases["sku"].isna().any()
    assert (cases["sku"] == "").any()


def test_load_table_reads_once():
    assert load_table("store_master") is load_table("store_master")


def test_window_is_open_at_start_and_closed_at_anchor():
    times = [
        config.ANCHOR - pd.Timedelta(hours=24),    # exactly 24h ago: outside
        config.ANCHOR - pd.Timedelta(hours=23),    # inside
        config.ANCHOR,                             # inside
        config.ANCHOR + pd.Timedelta(minutes=1),   # after anchor: outside
    ]
    table = pd.DataFrame({"created_ts": times})
    kept = within_window(table, "created_ts", hours=24)
    assert list(kept["created_ts"]) == times[1:3]


def test_stockout_with_consecutive_short_shipments_and_vendor_warning():
    result = data_agent.stockout_root_cause("ST002", "SKU1001")
    assert result.facts["stock_status"]["on_hand"] == 0
    assert result.facts["high_demand"] is True
    receipts = result.facts["receipt_history"]["receipts"]
    assert [(r["received_qty"], r["po_qty"]) for r in receipts[:2]] == [(18, 50), (20, 45)]
    assert result.facts["vendor_performance"]["fill_rate"] == 0.68
    for finding in ["stockout_high_demand", "short_shipment", "consecutive_short_shipment", "vendor_warning"]:
        assert finding in result.findings
    assert {"inventory.csv", "receipts.csv", "vendor_master.csv"} <= set(result.data_used)


def test_warehouse_depleted_when_store_and_warehouse_are_zero():
    result = data_agent.stockout_root_cause("ST005", "SKU7001")
    assert result.facts["stock_status"]["on_hand"] == 0
    warehouse = result.facts["warehouse_status"]
    assert warehouse["warehouse_on_hand"] == 0
    assert warehouse["latest_warehouse_receipt"]["receipt_id"] == "RCPT-017"
    assert warehouse["latest_warehouse_receipt"]["received_qty"] == 0
    assert "warehouse_depleted" in result.findings
    assert {"inventory.csv", "receipts.csv"} <= set(result.data_used)


def test_price_lookup_confirms_list_price_mismatch():
    result = data_agent.price_lookup("SKU2003", [899.0, 1099.0])
    assert result.facts["list_price_inr"] == 1099
    assert result.findings == ["price_mismatch_confirmed"]
    assert data_agent.price_lookup("SKU2003", [1099.0]).findings == []


def test_refund_thresholds():
    assert "refund_needs_approval" in data_agent.refund_check(24999, False).findings
    assert "refund_needs_approval" in data_agent.refund_check(1500, False).findings
    small_no_receipt = data_agent.refund_check(800, False)
    assert small_no_receipt.findings == ["no_receipt_review"]   # review, but below the approval threshold
    assert data_agent.refund_check(5000, True).findings == ["refund_needs_approval"]
    assert data_agent.refund_check(4999, True).findings == []


def test_case_query_uses_anchor_window_priorities_and_statuses():
    result = data_agent.case_query(["P1", "P2"], hours=24)
    cases = load_table("support_cases").set_index("case_id")
    case_ids = [case_id for group in result.facts["groups"] for case_id in group["case_ids"]]
    assert len(case_ids) == result.facts["case_count"] > 0
    for case_id in case_ids:
        case = cases.loc[case_id]
        assert config.ANCHOR - pd.Timedelta(hours=24) < case["created_ts"] <= config.ANCHOR
        assert case["priority_label"] in ("P1", "P2")
        assert case["status"] in ("Open", "In Progress")
    assert result.data_used[0] == "support_cases.csv"


def test_quality_complaints_for_mention_are_repeated_across_stores():
    result = data_agent.quality_complaints(mention="football")
    assert result.facts["store_count"] >= 2
    assert "repeated_quality_complaints" in result.findings
    assert {"returns_quality.csv", "support_cases.csv"} <= set(result.data_used)


def test_vendor_ranking_is_sorted_and_never_flags_safety():
    result = data_agent.vendor_quality_ranking(30)
    counts = [vendor["complaint_count"] for vendor in result.facts["ranking"]]
    assert counts == sorted(counts, reverse=True)
    assert "repeated_quality_complaints" in result.findings
    assert "safety_incident" not in result.findings
    assert {"returns_quality.csv", "vendor_master.csv"} <= set(result.data_used)


def test_tools_report_high_demand():
    assert data_agent.stock_status("ST005", "SKU7001").facts["high_demand"] is True
    assert data_agent.receipt_history("ST002", "SKU1001").facts["high_demand"] is True
    assert data_agent.price_lookup("SKU5002", []).facts["high_demand"] is False


def test_dispatcher_returns_no_data_when_store_is_missing():
    result = triage.triage(safety.check_input("Cricket Helmet Pro is out of stock, customers keep asking."), None, None)
    assert result.facts.sku == "SKU1004"
    assert "store_id" in result.missing_fields
    assert data_agent.run_data_agent(result) == []


def test_defect_mention_ignores_packaging_and_adds_quality_check_to_returns():
    assert triage.extract_defect_mentioned("Box arrived with damaged packaging only.") is False
    result = triage.triage(safety.check_input(
        "Refund request at ST001 with receipt: the Exercise Bike arrived damaged, INR 17999."), None, None)
    assert result.category == "Returns and Refunds"
    assert result.facts.defect_mentioned is True
    tools = [data_result.tool for data_result in data_agent.run_data_agent(result)]
    assert tools == ["refund_check", "quality_complaints"]


def test_false_availability_only_when_shelf_is_empty():
    assert "false_availability" in data_agent.stock_findings(0, 4, 10, True)
    shelf_has_stock = data_agent.stock_findings(12, 18, 15, True)
    assert "inventory_discrepancy" in shelf_has_stock and "false_availability" not in shelf_has_stock
