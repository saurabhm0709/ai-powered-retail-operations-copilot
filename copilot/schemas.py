"""Data contracts passed between agents and used for LLM structured outputs (ARCHITECTURE.md §5)."""

from typing import Literal

from pydantic import BaseModel

Category = Literal["POS and Payments", "Inventory and Replenishment", "Pricing and Promotions",
                   "Returns and Refunds", "Vendor and Supply Chain", "Product Quality and Safety",
                   "Customer Service", "Other or Security"]
Priority = Literal["P1", "P2", "P3", "P4"]
Scope = Literal["one_register", "multiple_registers", "all_registers", "one_sku", "many_skus",
                "department", "one_store", "multiple_stores", "many_customers", "unknown"]
SafetySignal = Literal["injury", "recall", "unsafe_equipment", "equipment_malfunction", "none"]
AnalyticsIntent = Literal["open_cases_window", "vendor_quality_ranking", "quality_trend", "short_shipments",
                          "stockout_root_cause", "returns_summary", "inventory_below_reorder", "other"]


# Input gate output: masked text plus whether the request is blocked and why.
class SafetyCheck(BaseModel):
    masked_text: str
    pii_found: list[str]            # e.g. ["phone","card"]
    blocked: bool
    threat_type: Literal["none", "prompt_injection", "restricted_data_request", "payment_data_request"]
    block_reason: str | None


# Facts extracted from the request; normalised and SKU-resolved by code.
class Facts(BaseModel):
    store_id: str | None            # normalised "ST002" / "WH001"
    product_mention: str | None
    sku: str | None                 # resolved by code (never by LLM)
    sku_candidates: list[str] = []
    register_id: str | None
    scope: Scope = "unknown"
    amount_inr: float | None
    receipt_present: bool | None
    safety_signal: SafetySignal = "none"
    order_context: Literal["academy", "school", "tournament", "team", "none"] = "none"
    repeated: bool = False          # "again", "repeated", "multiple complaints"
    multi_store: bool = False
    defect_mentioned: bool = False  # damaged/defective item (not packaging) -> also check quality complaints
    time_window_hours: int | None   # "last 24 hours" -> 24, "last 30 days" -> 720
    priority_filter: list[Priority] = []   # "P1 and P2 incidents" -> ["P1","P2"] (analytics case_query)
    quoted_prices_inr: list[float] = []    # every INR amount in order, e.g. shelf 899 then POS 1099 (price_lookup)


# Facts the Triage LLM reads from the text. No sku: code resolves the SKU from product_mention.
# No defaults: OpenAI strict structured output requires every field.
class LLMFacts(BaseModel):
    store_id: str | None
    product_mention: str | None
    register_id: str | None
    scope: Scope
    amount_inr: float | None
    receipt_present: bool | None
    safety_signal: SafetySignal
    order_context: Literal["academy", "school", "tournament", "team", "none"]
    repeated: bool
    multi_store: bool
    defect_mentioned: bool
    time_window_hours: int | None
    priority_filter: list[Priority]
    quoted_prices_inr: list[float]


# Triage LLM structured output. No sku, priority, calibrated confidence, routing or missing fields:
# code decides those after the call (ARCHITECTURE.md §6 [2]).
class TriageLLMOutput(BaseModel):
    mode: Literal["triage", "analytics", "policy_question"]
    category: Category
    llm_confidence: float
    analytics_intent: AnalyticsIntent | None
    facts: LLMFacts
    suggested_priority: Priority   # used only by PRIORITY_MODE=llm and the rules-vs-LLM comparison
    short_reasoning: str


# Triage Agent output: classification, routing and preliminary priority.
class TriageResult(BaseModel):     # LLM fills these via structured output (except sku/priority)
    mode: Literal["triage", "analytics", "policy_question"]
    category: Category
    llm_confidence: float
    confidence: float              # calibrated by rules.calibrate_confidence
    facts: Facts
    needs_policy: bool
    needs_data: bool
    analytics_intent: AnalyticsIntent | None
    preliminary_priority: Priority
    preliminary_rule_id: str
    missing_fields: list[str]
    short_reasoning: str
    llm_priority: Priority | None = None   # LLM's own suggestion; final priority is decided by rules


# One retrieved policy section; the citation is also the ChromaDB chunk id.
class PolicyHit(BaseModel):
    citation: str                  # "03_inventory_replenishment_policy.md#stock-outs-and-replenishment"
    file: str
    section: str
    source_id: str
    text: str
    score: float


# One Data Agent tool result; the only source of numbers in the answer.
class DataResult(BaseModel):
    tool: str
    facts: dict                    # numbers used in the answer come ONLY from here
    findings: list[str]            # finding codes, e.g. ["short_shipment","stockout_high_demand"]
    summary_lines: list[str]       # human-readable evidence lines
    data_used: list[str]           # CSV file names
    evidence_rows: list[dict] = [] # small, masked


# Controller escalation decision from the escalation matrix.
class Escalation(BaseModel):
    required: bool
    teams: list[str]
    team: str                      # team = " + ".join(unique teams)
    channel: str | None
    sla_minutes: int | None
    human_approval_required: bool
    reason: str


# Controller decision: final priority, escalation, citations and workflow payload (ARCHITECTURE.md §6 [4b]).
class Decision(BaseModel):
    priority: Priority
    rule_id: str
    priority_reason: str
    fired_rule_ids: list[str]      # winning rule first, then other rules that fired
    allowed_citations: list[str]   # the only ids the responder may cite
    citations: list[str]           # final list: rule citations first, then retrieved, max 4
    escalation: Escalation
    workflow_payload: dict | None


# Final Response LLM structured output; validated by safety.check_output before use.
class ResponseLLMOutput(BaseModel):
    answer: str
    next_actions: list[str]
    follow_up_question: str | None
    citations: list[str]


# Final response returned to the UI; maps 1:1 to a predictions.csv row.
class CopilotResponse(BaseModel):
    trace_id: str
    mode: str
    category: Category
    priority: Priority
    priority_rule_id: str
    priority_reason: str
    confidence: float
    answer: str
    next_actions: list[str]
    follow_up_question: str | None
    citations: list[str]
    data_used: list[str]
    data_findings: dict
    escalation: Escalation
    pii_masked: bool
    safety_refusal: bool
    workflow_payload: dict | None
    limitations: str | None
