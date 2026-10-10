"""Every tunable parameter for the copilot (ARCHITECTURE.md §4). Never hardcode these inside functions."""

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; real environment variables take precedence


# Reads an on/off flag from the environment ("1", "true", "yes" -> True).
def env_flag(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in ("1", "true", "yes")


# --- Time ---
# CSV timestamps are naive IST; rows after this moment are excluded (AGENTS.md rule 7).
ANCHOR = pd.Timestamp("2026-10-10 12:00:00")

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "knowledge_base"
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "evaluation"
OUT_DIR = PROJECT_ROOT / "outputs"
CHROMA_DIR = PROJECT_ROOT / ".chroma"
CACHE_DIR = PROJECT_ROOT / ".cache"
TRACES_PATH = OUT_DIR / "traces.jsonl"
TICKETS_PATH = OUT_DIR / "tickets.jsonl"  # mock workflow system

# --- LLM ---
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TEMPERATURE = 0
TRIAGE_MAX_TOKENS = 400
RESPONSE_MAX_TOKENS = 500
MAX_RETRIES = 2
RETRY_BACKOFF_SECONDS = 1.0  # wait 1s, then 2s between retries
MAX_CALLS = 300  # hard budget guard per process (≈ $0.15 at gpt-4o-mini prices)
PROMPT_VERSION = "v1"  # part of the cache key; bump when prompts change

# --- Flags ---
MOCK_LLM = env_flag("MOCK_LLM")
PRIORITY_MODE = os.getenv("PRIORITY_MODE", "rules")  # "rules" | "llm"; "llm" only for the evaluation comparison

# --- Policy thresholds ---
# 05_responsible_ai_policy.md#human-in-the-loop, 01_retail_operations_sop.md#escalation-policy
CONFIDENCE_ESCALATION = 0.60
# ARCHITECTURE.md §6 [2] step 4: the keyword classifier disagreeing on category is real doubt about the case.
CATEGORY_DISAGREE_PENALTY = 0.20
# Missing fields drive the follow-up question, not escalation: asking one question is cheaper than paging
# a manager; escalation = a human must decide. So they cost little (05_responsible_ai_policy.md#uncertainty).
MISSING_FIELD_PENALTY = 0.05
MISSING_FIELD_PENALTY_MAX = 0.10
# 02_pos_pricing_returns_policy.md#returns-and-refunds
REFUND_APPROVAL_ANY = 5000
# 02_pos_pricing_returns_policy.md#returns-and-refunds
REFUND_APPROVAL_NO_RECEIPT = 1000
# 04_vendor_quality_policy.md#short-shipments
SHORT_SHIPMENT_PCT = 0.90
# 03_inventory_replenishment_policy.md#inventory-discrepancy
DISCREPANCY_UNITS = 10
# 03_inventory_replenishment_policy.md#inventory-discrepancy
DISCREPANCY_PCT = 0.20
# 04_vendor_quality_policy.md#vendor-performance (warning signal, not proof)
VENDOR_FILL_WARN = 0.80
# 04_vendor_quality_policy.md#vendor-performance (warning signal, not proof)
VENDOR_ONTIME_WARN = 0.85
# 04_vendor_quality_policy.md#product-quality-complaints
QUALITY_WINDOW_DAYS = 30
# 04_vendor_quality_policy.md#product-quality-complaints ("repeated" = at least 2 complaints or 2 stores)
REPEATED_COMPLAINTS_MIN = 2

# --- Data Agent (tuning values, not policy) ---
WAREHOUSE_ID = "WH001"
RECEIPT_HISTORY_N = 3          # receipts checked per store+SKU (ARCHITECTURE.md §6 [3B])
RELATED_CASES_DAYS = 30        # window for related support cases
CASE_QUERY_DEFAULT_HOURS = 24  # analytics window when the question names none
CASE_QUERY_PRIORITIES = ["P1", "P2", "P3", "P4"]  # all priorities when the question names none
MAX_LIST_ITEMS = 5             # cap on ranked lists / evidence rows passed to the answer

# --- Retrieval (tuning values, not policy) ---
RAG_TOP_K = 4
RAG_MAX_DISTANCE = 0.65  # cosine distance cutoff; tune against the public tests
CHROMA_COLLECTION = "policies"
MAX_CITATIONS = 4  # ARCHITECTURE.md §6 [4b]: final citations list is capped at 4

# --- Mock mode ---
MOCK_LLM_CONFIDENCE = 0.7  # ARCHITECTURE.md §6 [2]: fixed llm_confidence for the keyword classifier
