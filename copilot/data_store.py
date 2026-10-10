"""Loads the provided CSVs once, with typed columns, plus anchor-time filters (ARCHITECTURE.md §3, §6 [3B])."""

from functools import lru_cache

import pandas as pd

from copilot import config

# Timestamp columns per table; parsed so time filters compare datetimes, not strings.
DATE_COLUMNS = {
    "inventory": ["last_count_date"],
    "receipts": ["receipt_date"],
    "returns_quality": ["created_ts"],
    "support_cases": ["created_ts"],
    "vendor_master": ["effective_from"],
}

# "Yes"/"No" columns converted to True/False so rule code can test them directly.
YES_NO_COLUMNS = {
    "escalation_matrix": ["human_approval_required"],
    "product_master": ["high_demand"],
    "receipts": ["quality_check_ok"],
    "returns_quality": ["receipt_present"],
}


# Table name (CSV file name without .csv) -> typed DataFrame. Read from disk only once per process;
# callers must not modify the returned frame (copy it first) because it is shared.
@lru_cache(maxsize=None)
def load_table(name: str) -> pd.DataFrame:
    table = pd.read_csv(config.DATA_DIR / f"{name}.csv")
    for column in DATE_COLUMNS.get(name, []):
        table[column] = pd.to_datetime(table[column])
    for column in YES_NO_COLUMNS.get(name, []):
        table[column] = table[column] == "Yes"
    # Blank text cells become "" so string filters (.str.contains) never meet NaN.
    # is_string_dtype works on pandas 2 (object) and pandas 3 (str) text columns.
    for column in table.columns:
        if pd.api.types.is_string_dtype(table[column]):
            table[column] = table[column].fillna("")
    return table


# Rows whose timestamp is at or before the anchor; later rows are excluded (AGENTS.md rule 7).
def before_anchor(table: pd.DataFrame, column: str) -> pd.DataFrame:
    return table[table[column] <= config.ANCHOR]


# Rows inside the window (ANCHOR - days - hours, ANCHOR], e.g. "last 24 hours" = hours=24.
def within_window(table: pd.DataFrame, column: str, days: int = 0, hours: int = 0) -> pd.DataFrame:
    window_start = config.ANCHOR - pd.Timedelta(days=days, hours=hours)
    in_window = (table[column] > window_start) & (table[column] <= config.ANCHOR)
    return table[in_window]
