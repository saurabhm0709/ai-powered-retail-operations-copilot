"""[2] Triage Agent: category, mode, facts and routing (ARCHITECTURE.md §6 [2]).

This step uses the keyword classifier + regex facts (the MOCK_LLM path). The LLM call is added in step 2.
"""

import re

from copilot import config, data_agent, rules
from copilot.schemas import AnalyticsIntent, Facts, TriageResult

# Analytics questions ask about many cases at once, not one live issue.
ANALYTICS_PATTERN = r"\b(how many|which vendor|which store|grouped by|group by|show (all|unresolved|open)|list (all|open)" \
                    r"|top \d+|trend|summar(y|ise|ize))\b"

# Checked in order; first match wins.
ANALYTICS_INTENTS: list[tuple[AnalyticsIntent, str]] = [
    ("vendor_quality_ranking", r"vendor.*(quality|complaint)|(quality|complaint).*vendor"),
    ("quality_trend", r"quality|complaint|defect"),
    ("short_shipments", r"short ship|received less"),
    ("returns_summary", r"return|refund"),
    ("inventory_below_reorder", r"reorder|below"),
    ("open_cases_window", r"incident|case|ticket|unresolved|open"),
]

SAFETY_SIGNALS = [("injury", r"injur|hurt|bleed"), ("recall", r"recall"),
                  ("unsafe_equipment", r"unsafe|hazard|sparking|electric shock"),
                  ("equipment_malfunction", r"malfunction|belt slip|stopped suddenly")]

# Critical fields per category (01_retail_operations_sop.md#minimum-triage-fields).
CRITICAL_FIELDS = {"Inventory and Replenishment": ["store_id", "sku"],
                   "Returns and Refunds": ["amount_inr", "receipt_present"],
                   "Pricing and Promotions": ["sku"]}

DATA_CATEGORIES = {"Inventory and Replenishment", "Vendor and Supply Chain",
                   "Product Quality and Safety", "Returns and Refunds"}


# Text + optional UI store -> "ST002" / "WH001" / None. Also accepts "store 2".
def extract_store_id(text: str, store_hint: str | None) -> str | None:
    match = re.search(r"\b(ST\d{3}|WH001)\b", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\bstore\s+(\d{1,3})\b", text, re.IGNORECASE)
    if match:
        return f"ST{int(match.group(1)):03d}"
    return store_hint


# Text -> "REG-04" from "register 4", "register #4" or "REG-04", else None.
def extract_register_id(text: str) -> str | None:
    match = re.search(r"\b(?:register|reg)\s*(?:no\.?|#|-)?\s*(\d{1,2})\b", text, re.IGNORECASE)
    return f"REG-{int(match.group(1)):02d}" if match else None


# Text -> first INR/Rs amount as float ("INR 24,999" -> 24999.0), else None.
def extract_amount(text: str) -> float | None:
    match = re.search(r"(?:INR|Rs\.?|₹)\s?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    return float(match.group(1).replace(",", "")) if match else None


# Text -> every INR/Rs amount in order ("shelf INR 899 but POS INR 1099" -> [899.0, 1099.0]).
def extract_all_amounts(text: str) -> list[float]:
    amounts = []
    for match in re.finditer(r"(?:INR|Rs\.?|₹)\s?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE):
        amounts.append(float(match.group(1).replace(",", "")))
    return amounts


# Text -> priority labels named in it ("P1 and P2 incidents" -> ["P1", "P2"]).
def extract_priority_filter(text: str) -> list[str]:
    labels = []
    for label in re.findall(r"\bP([1-4])\b", text, re.IGNORECASE):
        if f"P{label}" not in labels:
            labels.append(f"P{label}")
    return labels


# Text -> False for "no/without receipt", True for "with/has receipt", None if not mentioned.
def extract_receipt_present(text: str) -> bool | None:
    if re.search(r"\b(no|without|lost|missing)\s+(the\s+)?receipt", text, re.IGNORECASE):
        return False
    if re.search(r"\b(with|has|have)\s+(the\s+|a\s+)?receipt", text, re.IGNORECASE):
        return True
    return None


# Text -> impact scope (01_retail_operations_sop.md#priority-rules: one register vs many vs store-wide).
def extract_scope(text: str) -> str:
    lowered = text.lower()
    if re.search(r"\b(all|every) (the )?registers?\b|no register", lowered):
        return "all_registers"
    if re.search(r"store-?wide|whole store|entire store", lowered):
        return "one_store"
    if re.search(r"\b(multiple|several|two|three|many) registers\b|registers \d+ and \d+", lowered):
        return "multiple_registers"
    if re.search(r"\bregister\b", lowered):
        return "one_register"
    if re.search(r"\b(two|multiple|several|many|all) stores\b", lowered):
        return "multiple_stores"
    return "unknown"


# Text -> safety signal code, "none" if no safety words.
def extract_safety_signal(text: str) -> str:
    for signal, pattern in SAFETY_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            return signal
    return "none"


# Text -> time window in hours ("last 24 hours" -> 24, "last 30 days" -> 720), else None.
def extract_time_window(text: str) -> int | None:
    match = re.search(r"last\s+(\d+)\s+(hour|day)s?", text, re.IGNORECASE)
    if not match:
        return None
    amount = int(match.group(1))
    return amount if match.group(2).lower() == "hour" else amount * 24


# Text -> order context ("academy order" -> "academy"), "none" if absent.
def extract_order_context(text: str) -> str:
    match = re.search(r"\b(academy|school|tournament|team)\b", text, re.IGNORECASE)
    return match.group(1).lower() if match else "none"


# Masked text + UI store hint -> Facts, all by regex and code (no LLM).
def extract_facts(text: str, store_hint: str | None) -> Facts:
    mention = data_agent.find_product_mention(text)
    sku, sku_candidates = data_agent.resolve_sku(mention) if mention else (None, [])
    scope = extract_scope(text)
    return Facts(
        store_id=extract_store_id(text, store_hint),
        product_mention=mention,
        sku=sku, sku_candidates=sku_candidates,
        register_id=extract_register_id(text), scope=scope,
        amount_inr=extract_amount(text), receipt_present=extract_receipt_present(text),
        safety_signal=extract_safety_signal(text), order_context=extract_order_context(text),
        repeated=bool(re.search(r"\b(again|repeated|repeatedly|keeps|multiple complaints)\b", text, re.IGNORECASE)),
        multi_store=scope == "multiple_stores",
        time_window_hours=extract_time_window(text),
        priority_filter=extract_priority_filter(text), quoted_prices_inr=extract_all_amounts(text),
    )


# Masked text -> (mode, analytics intent or None).
def detect_mode(text: str) -> tuple[str, AnalyticsIntent | None]:
    lowered = text.lower()
    if not re.search(ANALYTICS_PATTERN, lowered):
        return "triage", None
    for intent, pattern in ANALYTICS_INTENTS:
        if re.search(pattern, lowered):
            return "analytics", intent
    return "analytics", "other"


# Category + facts -> list of missing critical field names (drives the follow-up question).
def find_missing_fields(category: str, facts: Facts) -> list[str]:
    missing = []
    for field in CRITICAL_FIELDS.get(category, []):
        if getattr(facts, field) is None:
            missing.append(field)
    # An ambiguous product needs a follow-up in every category (ARCHITECTURE.md §6 [2] step 3).
    if facts.sku_candidates and "sku" not in missing:
        missing.append("sku")
    return missing


# ARCHITECTURE.md §6 [2] step 6: when Data Agent tools are worth running.
def needs_data_for(mode: str, category: str, facts: Facts) -> bool:
    if mode == "analytics" or category in DATA_CATEGORIES:
        return True
    return category == "Pricing and Promotions" and (facts.sku is not None or facts.store_id is not None)


# SafetyCheck masked text + persona + store hint -> TriageResult (keyword path; LLM path in step 2).
def triage(safety, persona: str | None, store_hint: str | None) -> TriageResult:
    text = safety.masked_text
    mode, analytics_intent = detect_mode(text)
    category = rules.keyword_category(text)
    facts = extract_facts(text, store_hint)
    missing_fields = find_missing_fields(category, facts)
    # In the keyword path the "LLM" category IS the keyword category, so only missing fields lower confidence.
    confidence = rules.calibrate_confidence(config.MOCK_LLM_CONFIDENCE, category, category, missing_fields)
    result = TriageResult(
        mode=mode, category=category,
        llm_confidence=config.MOCK_LLM_CONFIDENCE, confidence=confidence,
        facts=facts, needs_policy=True, needs_data=needs_data_for(mode, category, facts),
        analytics_intent=analytics_intent, preliminary_priority="P3", preliminary_rule_id="P3-DEFAULT",
        missing_fields=missing_fields,
        short_reasoning=f"Keyword classifier matched '{category}'.",
    )
    priority, rule_id = rules.decide_priority(result, data_results=[])
    return result.model_copy(update={"preliminary_priority": priority, "preliminary_rule_id": rule_id})
