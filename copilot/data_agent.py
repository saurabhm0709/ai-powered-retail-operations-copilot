"""[3B] Data Agent: pandas tools over the CSVs + deterministic dispatcher (ARCHITECTURE.md §6 [3B])."""

import difflib
import re
from functools import lru_cache

import pandas as pd

from copilot.data_store import load_table
from copilot.schemas import DataResult, TriageResult

MAX_SKU_CANDIDATES = 3
# Words that do not identify a product: the store sells "sports goods", "Size 5" is a size, "set"/"pair" a quantity.
SKU_STOPWORDS = {"the", "and", "for", "with", "sports", "goods", "item", "product", "size", "set", "pair"}
TYPO_CUTOFF = 0.8  # difflib similarity needed to correct a misspelt word ("tenis" -> "tennis")


# Lowercase words of 3+ letters minus stopwords (digits dropped, so "Size 5" does not need to be typed).
def name_words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{3,}", text.lower())) - SKU_STOPWORDS


# All words used in product names (the vocabulary for mention finding and typo correction).
@lru_cache(maxsize=1)
def product_vocabulary() -> frozenset[str]:
    vocabulary = set()
    for name in load_table("product_master")["name"]:
        vocabulary |= name_words(name)
    return frozenset(vocabulary)


# Free text -> longest run of consecutive product-name words ("Tennis Ball Pack"), or None.
# Used by the keyword (mock) triage; the LLM triage extracts product_mention itself.
def find_product_mention(text: str) -> str | None:
    vocabulary = product_vocabulary()
    best_run, current_run = [], []
    for word in re.findall(r"[A-Za-z]+", text):
        if word.lower() in vocabulary:
            current_run.append(word)
            if len(current_run) > len(best_run):
                best_run = list(current_run)
        else:
            current_run = []
    return " ".join(best_run) if best_run else None


# Products whose name contains ALL the given words.
def products_with_all_words(words: set[str]) -> pd.DataFrame:
    products = load_table("product_master")
    contains_all = products["name"].apply(lambda name: words <= name_words(name))
    return products[contains_all]


# Mention words -> same words with typos fixed to the closest product-name word (difflib, stdlib).
def correct_typos(words: set[str]) -> set[str]:
    corrected = set()
    for word in words:
        close = difflib.get_close_matches(word, product_vocabulary(), n=1, cutoff=TYPO_CUTOFF)
        corrected.add(close[0] if close else word)
    return corrected


# Matching products -> the one core SKU (not an SKU8xxx variant) whose whole name the mention covers, else None.
# "Match Football" covers core "Match Football Size 5"; "running" does not cover "Running Shoes" -> ambiguous.
def covered_core_sku(matches: pd.DataFrame, words: set[str]) -> str | None:
    core = matches[~matches["sku"].str.startswith("SKU8")]
    covered = core[core["name"].apply(lambda name: name_words(name) <= words)]
    return covered["sku"].iloc[0] if len(covered) == 1 else None


# Product mention -> (sku or None, candidate skus). ARCHITECTURE.md §6 [2] step 3: exact name, else all
# mention words in the name (typo fallback only if nothing matches), prefer core, else up to 3 candidates.
def resolve_sku(mention: str) -> tuple[str | None, list[str]]:
    products = load_table("product_master")
    exact = products[products["name"].str.lower() == mention.strip().lower()]
    if len(exact) == 1:
        return exact["sku"].iloc[0], []
    words = name_words(mention)
    if not words:
        return None, []
    matches = products_with_all_words(words)
    if len(matches) == 0:
        words = correct_typos(words)
        matches = products_with_all_words(words)
    if len(matches) == 1:
        return matches["sku"].iloc[0], []
    core_sku = covered_core_sku(matches, words)
    if core_sku:
        return core_sku, []
    # Core SKUs first so the follow-up question offers the main product before its variants.
    ranked = matches.assign(is_variant=matches["sku"].str.startswith("SKU8")).sort_values(["is_variant", "sku"])
    return None, ranked["sku"].head(MAX_SKU_CANDIDATES).tolist()


# SKU -> product name from product_master.
def product_name(sku: str) -> str:
    products = load_table("product_master")
    return products[products["sku"] == sku]["name"].iloc[0]


# Triage -> data tool results. STUB: returns no data until the tools are built (step 2).
def run_data_agent(triage: TriageResult) -> list[DataResult]:
    return []
