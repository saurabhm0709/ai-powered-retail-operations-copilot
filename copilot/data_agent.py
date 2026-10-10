"""[3B] Data Agent: pandas tools over the CSVs + deterministic dispatcher (ARCHITECTURE.md §6 [3B])."""

import re

from copilot.data_store import load_table
from copilot.schemas import DataResult, TriageResult

MAX_SKU_CANDIDATES = 3


# Lowercase words of 3+ letters (digits dropped, so "Size 5" does not need to be typed).
def name_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower()))


# Product mention (or whole masked text in mock mode) -> (sku or None, candidate skus).
# Picks products sharing the most words with the mention; core SKUs (not SKU8xxx variants) are preferred.
def resolve_sku(mention: str) -> tuple[str | None, list[str]]:
    products = load_table("product_master")
    mention_words = name_words(mention)
    overlaps = products["name"].apply(lambda name: len(name_words(name) & mention_words))
    best_overlap = overlaps.max()
    if best_overlap == 0:
        return None, []
    best = products[overlaps == best_overlap]
    core = best[~best["sku"].str.startswith("SKU8")]
    if len(core) > 0:
        best = core
    if len(best) == 1:
        return best["sku"].iloc[0], []
    return None, best["sku"].head(MAX_SKU_CANDIDATES).tolist()


# SKU -> product name from product_master.
def product_name(sku: str) -> str:
    products = load_table("product_master")
    return products[products["sku"] == sku]["name"].iloc[0]


# Triage -> data tool results. STUB: returns no data until the tools are built (step 2).
def run_data_agent(triage: TriageResult) -> list[DataResult]:
    return []
