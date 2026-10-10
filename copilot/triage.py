"""[2] Triage Agent: category, mode, facts and routing (ARCHITECTURE.md §6 [2]).

The LLM reads the wording (category, mode, facts). Code then normalises facts, resolves the SKU, calibrates
confidence and sets the preliminary priority. MOCK_LLM or an LLM failure uses the keyword classifier instead.
"""

import json
import re

from copilot import config, data_agent, llm, rules
from copilot.schemas import AnalyticsIntent, Facts, LLMFacts, TriageLLMOutput, TriageResult

# Categories and examples from 01_retail_operations_sop.md#service-categories. Few-shot examples are our own
# and deliberately use different products, stores and wording from evaluation/public_test_cases.csv.
TRIAGE_SYSTEM_PROMPT = """You triage internal requests from staff at Punjab Sports Mart, a sports goods retailer.
Read the message and return the structured fields. You only classify and extract; you never decide priority.

CATEGORIES (pick exactly one):
- POS and Payments: registers, tills, card readers, UPI or card payment failures, POS errors.
- Inventory and Replenishment: stock-outs, "not available", on-hand vs system counts, reorder, transfers.
- Pricing and Promotions: shelf label vs POS price, promotions, discounts.
- Returns and Refunds: returns, exchanges, refunds, receipts.
- Vendor and Supply Chain: deliveries, received less than ordered, short shipments, purchase orders.
- Product Quality and Safety: defects (seams, cracks, warping, tears), injuries, recalls, unsafe or malfunctioning equipment.
- Customer Service: complaints about service, waiting, staff behaviour, loyalty.
- Other or Security: general case lists or counts across categories, anything else.

RULES:
- If a request spans categories, choose the category linked to the immediate risk.
- A stock-out, "not available" or "out of stock" is Inventory and Replenishment even if the cause may be a vendor.
- Receiving less than was ordered (a delivery event) is Vendor and Supply Chain.
- Defects, cracks, split seams, injury, recall or unsafe equipment are Product Quality and Safety.
- Questions asking for lists, counts, rankings or trends are mode "analytics". If about cases in general,
  category "Other or Security"; if about one domain (e.g. vendor quality complaints), that domain's category.
- A question about what the policy says, with no live incident, is mode "policy_question". Otherwise "triage".
- Customer phone/card details inside a return request do not change the category: still Returns and Refunds.
- Masked values like ******3210 or [REDACTED_CARD] are already safe; ignore them.

FACTS (null / "none" / false / [] when not stated; never guess):
- store_id as "ST###" or "WH001"; register_id as written; product_mention = the product words as written.
- scope: one_register (others work), multiple_registers, all_registers, one_store (store-wide), one_sku,
  many_skus, department, multiple_stores, many_customers, or unknown.
- amount_inr: the refund/transaction amount; quoted_prices_inr: every INR price in the order written.
- receipt_present: false for "no/without receipt", true if the receipt or bill is available.
- safety_signal: injury, recall, unsafe_equipment, equipment_malfunction (e.g. demo machine failing), else none.
  A defect alone (split seam, crack) is "none" unless someone is hurt or the item is unsafe to use.
- repeated: "again", "repeated", several complaints. multi_store: two or more stores affected.
- defect_mentioned: the item itself is damaged or defective (damaged packaging alone is false).
- pos_severity (payments only): outage = registers cannot process / are down / not working;
  degraded = slow, intermittent, sometimes failing, timing out; unknown if not stated or not a POS issue.
- time_window_hours: "last 24 hours" = 24, "last 30 days" = 720. priority_filter: P1-P4 labels named.
- analytics_intent: only in analytics mode, else null.
- llm_confidence: 0-1, how sure you are of the category.

PRIORITY (suggested_priority, your own estimate from 01_retail_operations_sop.md#priority-rules; code decides
the final priority):
- P1: immediate human escalation: store-wide POS/payment outage, payment data or credential request, product
  safety or recall risk, equipment injury risk, unsafe fitness machine, active customer safety issue.
- P2: high impact, no immediate danger: repeated stock-out of high-demand items, vendor short shipment affecting
  replenishment, pricing error affecting many customers, multiple registers degraded, repeated quality
  complaints, urgent manager review (e.g. refund above approval threshold).
- P3: routine store issue: one register or card reader down, one SKU below reorder point, single shelf label
  mismatch, standard return, one-store inventory discrepancy.
- P4: general inquiry: policy questions, reporting requests, guidance with no active issue.

EXAMPLES (input -> key fields):
1. "Barcode scanner at till 2 in ST014 freezes after each scan; the other tills are fine."
   -> mode triage, category POS and Payments, store_id ST014, register_id "2", scope one_register.
2. "System shows 12 Hockey Stick Senior at Store 9 but we counted none on the rack."
   -> mode triage, category Inventory and Replenishment, store_id ST009, product_mention "Hockey Stick Senior".
3. "A shopper cut their hand on a cracked Cricket Helmet Pro at ST021."
   -> mode triage, category Product Quality and Safety, safety_signal injury, defect_mentioned true.
4. "Only 40 of 100 Feather Shuttlecock Tubes arrived at ST011 this morning."
   -> mode triage, category Vendor and Supply Chain, store_id ST011, product_mention "Feather Shuttlecock Tube".
5. "How many no-receipt returns did each store log in the last 30 days?"
   -> mode analytics, category Returns and Refunds, analytics_intent returns_summary, time_window_hours 720.
6. "Customer at ST007 wants to swap Gym Training Gloves for a bigger size, has the bill, INR 799."
   -> mode triage, category Returns and Refunds, amount_inr 799, receipt_present true.
"""

# Analytics questions ask about many cases at once, not one live issue.
ANALYTICS_PATTERN = r"\b(how many|which vendor|which store|grouped by|group by|show (all|unresolved|open)|list (all|open)" \
                    r"|top \d+|trend|summar(y|ise|ize))\b"
# Asking what the policy says, with no live incident (01_retail_operations_sop.md#priority-rules: P4).
POLICY_QUESTION_PATTERN = r"\b(what is|what's|what does) (our|the) (policy|rule|process)\b|\bpolicy (on|for)\b"

# Checked in order; first match wins.
ANALYTICS_INTENTS: list[tuple[AnalyticsIntent, str]] = [
    ("vendor_quality_ranking", r"vendor.*(quality|complaint)|(quality|complaint).*vendor"),
    ("quality_trend", r"quality|complaint|defect"),
    ("short_shipments", r"short ship|received less"),
    ("returns_summary", r"return|refund"),
    ("inventory_below_reorder", r"reorder|below"),
    ("open_cases_window", r"incident|case|ticket|unresolved|open"),
]

# 04_vendor_quality_policy.md#recall-and-safety. "cut" and "stuck" need a body word, so "price cut",
# "power cut" and "card stuck in reader" do not count as injuries.
SAFETY_SIGNALS = [("injury", r"injur|hurt|bleed|\bcut (?:his|her|their|my|a)\b"
                             r"|\b(?:finger|hand|toe|foot|arm|leg)s? (?:got )?(?:stuck|trapped|caught)"),
                  ("recall", r"recall"),
                  ("unsafe_equipment", r"unsafe|hazard|sparking|electric shock|\bsmok(?:e|ing)\b|burning smell"
                                       r"|smell of burning|\b(?:on|caught) fire\b|wobbl|(?:might|could|about to) fall"),
                  ("equipment_malfunction", r"malfunction|belt slip|stopped suddenly")]
# Amount formats seen in store messages, tried in order: "INR 1,499" / "Rs 18k", "1499 rs" / "1499/-",
# "amt 1499", then a bare "18k" (staff shorthand for thousands). Group 2 is the optional "k".
AMOUNT_PATTERNS = [r"(?:INR|Rs\.?|₹)\s?(\d[\d,]*(?:\.\d+)?)\s?(k)?\b",
                   r"\b(\d[\d,]*(?:\.\d+)?)\s?(k)?\s?(?:INR|Rs\b|rupees|/-)",
                   r"\b(?:amt|amount|worth)\s*(?:of|is|was|:|=)?\s*(\d[\d,]*(?:\.\d+)?)\s?(k)?\b",
                   r"\b(\d+(?:\.\d+)?)\s?(k)\b"]

# Critical fields per category (01_retail_operations_sop.md#minimum-triage-fields).
CRITICAL_FIELDS = {"Inventory and Replenishment": ["store_id", "sku"],
                   "Returns and Refunds": ["amount_inr", "receipt_present"],
                   "Pricing and Promotions": ["sku"]}
# Categories whose Data Agent tools need one SKU (stock, receipts, price, quality complaints for that product).
SKU_FOLLOW_UP_CATEGORIES = {"Inventory and Replenishment", "Pricing and Promotions", "Vendor and Supply Chain",
                            "Product Quality and Safety"}

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


# Text -> first amount as float ("INR 24,999" -> 24999.0, "amt 1499" -> 1499.0, "18k" -> 18000.0), else None.
def extract_amount(text: str) -> float | None:
    for pattern in AMOUNT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount = float(match.group(1).replace(",", ""))
            return amount * 1000 if match.group(2) else amount
    return None


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
    if re.search(r"\b(all|every) (the )?(registers?|tills?|counters?)\b|no register", lowered):
        return "all_registers"
    if re.search(r"store-?wide|whole store|entire store", lowered):
        return "one_store"
    if re.search(r"\b(multiple|several|two|three|many) (registers|tills|counters)\b|registers \d+ and \d+", lowered):
        return "multiple_registers"
    if re.search(r"\b(register|reg)\b", lowered):
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


# Damaged packaging alone is not a product defect (same exclusion as the quality complaint data).
DEFECT_PATTERN = r"\b(damaged(?!\s+packaging)|defect\w*|broken|torn|crack\w*|split|seams?|faulty)"


# Text -> True when the item itself is reported damaged or defective.
def extract_defect_mentioned(text: str) -> bool:
    return bool(re.search(DEFECT_PATTERN, text, re.IGNORECASE))


# 02_pos_pricing_returns_policy.md#pos-card-reader-runbook: "cannot process card payments" = outage (P1);
# 01_retail_operations_sop.md#priority-rules: "multiple registers degraded" = P2.
POS_OUTAGE_PATTERN = (r"\b(cannot|can't|can not|unable to)\b|\bdown\b|not working|offline|not going through"
                      r"|frozen|freez|stopped working|not responding")
POS_DEGRADED_PATTERN = r"\bslow|intermittent|sometimes|timing out|timed? out|\blag"


# Text -> "outage" / "degraded" / "unknown". Outage wins when both appear (fail safe towards P1).
def extract_pos_severity(text: str) -> str:
    if re.search(POS_OUTAGE_PATTERN, text, re.IGNORECASE):
        return "outage"
    if re.search(POS_DEGRADED_PATTERN, text, re.IGNORECASE):
        return "degraded"
    return "unknown"


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
        multi_store=scope == "multiple_stores", defect_mentioned=extract_defect_mentioned(text),
        pos_severity=extract_pos_severity(text),
        time_window_hours=extract_time_window(text),
        priority_filter=extract_priority_filter(text), quoted_prices_inr=extract_all_amounts(text),
    )


# Masked text -> (mode, analytics intent or None).
def detect_mode(text: str) -> tuple[str, AnalyticsIntent | None]:
    lowered = text.lower()
    if re.search(POLICY_QUESTION_PATTERN, lowered):
        return "policy_question", None
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
    # An ambiguous product needs a follow-up only where the SKU drives the data checks (ARCHITECTURE.md §6 [2]
    # steps 3-4). A return with amount and receipt status is decided without it, so it must not ask.
    if facts.sku_candidates and category in SKU_FOLLOW_UP_CATEGORIES and "sku" not in missing:
        missing.append("sku")
    return missing


# ARCHITECTURE.md §6 [2] step 6: when Data Agent tools are worth running.
def needs_data_for(mode: str, category: str, facts: Facts) -> bool:
    if mode == "analytics" or category in DATA_CATEGORIES:
        return True
    return category == "Pricing and Promotions" and (facts.sku is not None or facts.store_id is not None)


# Classification fields -> TriageResult with code-decided routing, missing fields and preliminary priority.
# Shared by the LLM path and the keyword path so both are finished by the same rules.
def finish_triage(mode: str, category: str, llm_confidence: float, keyword_cat: str, facts: Facts,
                  analytics_intent: AnalyticsIntent | None, short_reasoning: str) -> TriageResult:
    missing_fields = find_missing_fields(category, facts)
    confidence = rules.calibrate_confidence(llm_confidence, category, keyword_cat, missing_fields)
    result = TriageResult(
        mode=mode, category=category, llm_confidence=llm_confidence, confidence=confidence,
        facts=facts, needs_policy=True, needs_data=needs_data_for(mode, category, facts),
        analytics_intent=analytics_intent, preliminary_priority="P3", preliminary_rule_id="P3-DEFAULT",
        missing_fields=missing_fields, short_reasoning=short_reasoning,
    )
    priority, rule_id = rules.decide_priority(result, data_results=[])
    return result.model_copy(update={"preliminary_priority": priority, "preliminary_rule_id": rule_id})


# Masked text + store hint -> TriageResult from the keyword classifier and regex facts (MOCK_LLM / LLM failure).
def keyword_triage(text: str, store_hint: str | None) -> TriageResult:
    mode, analytics_intent = detect_mode(text)
    category = rules.keyword_category(text)
    # The "LLM" category IS the keyword category here, so only missing fields lower the confidence.
    return finish_triage(mode, category, config.MOCK_LLM_CONFIDENCE, category, extract_facts(text, store_hint),
                         analytics_intent, f"Keyword classifier matched '{category}'.")


# LLM register value ("4", "register 4", "REG-04") -> "REG-04", else None.
def normalize_register(value: str | None) -> str | None:
    match = re.search(r"\d{1,2}", value or "")
    return f"REG-{int(match.group()):02d}" if match else None


# LLM mention first, then the regex mention -> (mention, sku, candidates). SKU is always resolved by code.
def resolve_product(llm_mention: str | None, text: str) -> tuple[str | None, str | None, list[str]]:
    for mention in (llm_mention, data_agent.find_product_mention(text)):
        if mention:
            sku, candidates = data_agent.resolve_sku(mention)
            if sku or candidates:
                return mention, sku, candidates
    return llm_mention, None, []


# Code value if it found one, else the LLM value. Regex facts are exact; the LLM fills paraphrase gaps.
def code_first(code_value, llm_value, unknown=None):
    return llm_value if code_value == unknown else code_value


# Regex + LLM POS severity -> one value. The regex may only escalate: a false P2 on a real outage (a store
# that cannot take payments waits hours) costs far more than a false P1 on slow tills (a human downgrades it).
# So regex "outage" always wins; regex "degraded" is used only when the LLM could not tell.
def merge_pos_severity(code_severity: str, llm_severity: str) -> str:
    if code_severity == "outage":
        return "outage"
    if llm_severity != "unknown":
        return llm_severity
    return code_severity


# Regex facts + LLM facts -> one normalised Facts. Store: text regex > LLM > UI hint.
def merge_facts(text: str, llm_facts: LLMFacts, store_hint: str | None) -> Facts:
    code_facts = extract_facts(text, None)
    llm_store = extract_store_id(llm_facts.store_id or "", None)
    mention, sku, candidates = resolve_product(llm_facts.product_mention, text)
    return code_facts.model_copy(update={
        "store_id": code_facts.store_id or llm_store or store_hint,
        "product_mention": mention, "sku": sku, "sku_candidates": candidates,
        "register_id": code_first(code_facts.register_id, normalize_register(llm_facts.register_id)),
        "scope": code_first(code_facts.scope, llm_facts.scope, "unknown"),
        "amount_inr": code_first(code_facts.amount_inr, llm_facts.amount_inr),
        "receipt_present": code_first(code_facts.receipt_present, llm_facts.receipt_present),
        "safety_signal": code_first(code_facts.safety_signal, llm_facts.safety_signal, "none"),
        "order_context": code_first(code_facts.order_context, llm_facts.order_context, "none"),
        "repeated": code_facts.repeated or llm_facts.repeated,
        "multi_store": code_facts.multi_store or llm_facts.multi_store,
        "defect_mentioned": code_facts.defect_mentioned or llm_facts.defect_mentioned,
        "pos_severity": merge_pos_severity(code_facts.pos_severity, llm_facts.pos_severity),
        "time_window_hours": code_first(code_facts.time_window_hours, llm_facts.time_window_hours),
        "priority_filter": code_facts.priority_filter or llm_facts.priority_filter,
        "quoted_prices_inr": code_facts.quoted_prices_inr or llm_facts.quoted_prices_inr,
    })


# LLM output + masked text -> TriageResult. Confidence is calibrated against the keyword classifier.
def triage_from_llm(output: TriageLLMOutput, text: str, store_hint: str | None) -> TriageResult:
    facts = merge_facts(text, output.facts, store_hint)
    analytics_intent = output.analytics_intent
    if output.mode == "analytics" and analytics_intent is None:
        analytics_intent = detect_mode(text)[1] or "other"
    llm_confidence = max(0.0, min(1.0, output.llm_confidence))
    result = finish_triage(output.mode, output.category, llm_confidence, rules.keyword_category(text), facts,
                           analytics_intent, output.short_reasoning)
    return result.model_copy(update={"llm_priority": output.suggested_priority})


# Masked text + persona + UI store -> the user message for the Triage LLM (masked text only).
def triage_user_message(text: str, persona: str | None, store_hint: str | None) -> str:
    return json.dumps({"persona": persona or "Unknown", "selected_store": store_hint, "message": text},
                      ensure_ascii=False)


# SafetyCheck + persona + store hint -> TriageResult. LLM first; keyword path in MOCK_LLM or on LLM failure.
def triage(safety, persona: str | None, store_hint: str | None) -> TriageResult:
    text = safety.masked_text
    output, _usage = llm.structured(TRIAGE_SYSTEM_PROMPT, triage_user_message(text, persona, store_hint),
                                   TriageLLMOutput, config.TRIAGE_MAX_TOKENS)
    if output is None:
        return keyword_triage(text, store_hint)
    return triage_from_llm(output, text, store_hint)
