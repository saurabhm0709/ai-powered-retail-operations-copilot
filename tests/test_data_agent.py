"""Tests for data_store.py (and later data_agent.py). Uses the provided read-only CSVs."""

import pandas as pd

from copilot import config
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
