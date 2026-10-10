# Retail Operations Copilot — Punjab Sports Mart

An internal assistant for Punjab Sports Mart store staff (cashiers, store managers, inventory associates, customer
service and regional managers). An employee describes an issue or asks a question in plain text. The copilot masks
any personal or payment data, classifies the request into one of eight service categories, checks the store data
(inventory, receipts, vendors, returns, support cases), retrieves the relevant policy sections, and decides priority
and escalation with explicit rules that cite their policy source. It then writes a short answer with next actions
and citations, and a mock ticket for the workflow system. It recommends; it never approves refunds, submits vendor
claims, quarantines stock or declares equipment safe. Built for the HCLTech AI Engineer hackathon; the anchor time
for "today / last 24 hours / last 30 days" is `2026-10-10 12:00:00` IST.

## Setup and commands

```
pip install -r requirements.txt
python scripts/build_index.py            # build ChromaDB index once (--rebuild to force)
MOCK_LLM=1 python -m copilot.cli "All registers in ST003 cannot process cards"
pytest -q
python scripts/run_eval.py [--extra] [--priority-mode llm]
streamlit run app.py
```

- `MOCK_LLM=1` runs the full pipeline with zero network calls (keyword classifier + template answers). All tests
  use mock mode or fake LLM responses.
- Live mode uses OpenAI `gpt-4o-mini`: set `OPENAI_API_KEY` and `MOCK_LLM=0` in your environment. Every call is
  cached on disk, so re-running the same input is free, and a hard call budget (`MAX_CALLS` in `copilot/config.py`)
  stops a run that would overspend.
- Embeddings are local (ChromaDB's default `all-MiniLM-L6-v2`), so building the index costs nothing.
- `run_eval.py --out-dir DIR` writes results to another folder, so a trial run never overwrites `outputs/`.

## Architecture

> **The LLM interprets wording and writes the response. Python rules and source data control calculations, priority/escalation, citations and approval boundaries. The copilot recommends; it never approves refunds, submits vendor claims, or declares equipment safe.**

```
 Employee (cashier / store mgr / inventory / CS / regional mgr)
        |
        v
 [1] STREAMLIT UI  (app.py) ------------------------------------------------------------+
        |  free text + optional persona/store                                           |
        v                                                                               |
 [4a] SAFETY & WORKFLOW CONTROLLER — INPUT GATE  (deterministic)                        |
        - mask PII/payment data BEFORE any LLM sees the text                           |
        - detect prompt injection / restricted-data / payment-data requests            |
        |-- blocked --> refusal + P1 + Security escalation -------------------------+   |
        v                                                                           |   |
 [2] TRIAGE AGENT  (LLM structured output + Python rules)                           |   |
        - mode (triage | analytics | policy_question), category, confidence        |   |
        - extracted facts (store, product, register, scope, amount, receipt, ...)   |   |
        - SKU resolved by code; preliminary priority by rules                      |   |
        - routing: needs_policy? needs_data? analytics_intent?                      |   |
        |                                                                           |   |
        +-----------------------------+                                             |   |
        v                             v                                             |   |
 [3A] RAG AGENT                  [3B] DATA AGENT                                    |   |
   ChromaDB + local                 Pandas tools over the 9 CSVs                    |   |
   all-MiniLM-L6-v2 embeddings                                                      |   |
   policy chunks = KB sections      stock / receipts / vendor / quality /           |   |
   chunk id == citation             cases / refund / price / root-cause             |   |
        |                             |                                             |   |
        +-------------+---------------+                                             |   |
                      v                                                             |   |
 [4b] SAFETY & WORKFLOW CONTROLLER — DECISION  (deterministic)                      |   |
        - FINAL priority = rules(facts + data findings)                             |   |
        - escalation_matrix lookup -> team(s), channel, SLA, approval               |   |
        - escalate if P1 / confidence < 0.60 / restricted / approval required       |   |
        - allowed citation set = rule citations + finding citations + retrieved     |   |
        - mock workflow payload (ticket / escalation / vendor-claim draft)          |   |
                      v                                                             |   |
 [5] FINAL RESPONSE  (LLM drafts from evidence only) <------------------------------+   |
        - answer, next_actions, follow_up_question                                      |
        - OUTPUT GATE: citations ⊆ allowed set, re-mask PII, block approval language,   |
          template fallback if LLM fails                                                |
                      v                                                                 |
        CopilotResponse JSON  ------------------------------------------------------>  UI
        + outputs/traces.jsonl (audit)  + outputs/tickets.jsonl (mock workflow)
```

Each stage is one plain-function module in `copilot/` (`safety.py`, `triage.py`, `rag.py`, `data_agent.py`,
`controller.py`, `responder.py`, `workflow.py`, `orchestrator.py`) and passes typed Pydantic contracts
(`schemas.py`) to the next. Full spec: `ARCHITECTURE.md`.

## Metrics

All figures below come from `outputs/metrics.md` (real LLM, `gpt-4o-mini`, priority mode `rules`) or from the
reported runs on the unseen cases. That `metrics.md` run was made before the unseen-case fixes (commit `a99d0f3`
onwards) and has not been re-run since.

### Public test cases (11) — real LLM, rules mode

| Metric | Value | Target |
|---|---|---|
| Category accuracy | 11/11 (100%) | ≥ 75% |
| Priority accuracy | 11/11 (100%) | ≥ 75% |
| Escalation accuracy | 4/4 (100%) — only the 4 cases where ARCHITECTURE.md §9 states an expectation | report |
| Citation presence | 10/10 (100%) | ≥ 85% |
| Citation validity | 10/10 (100%) | 100% |
| Citation recall | 100% | report |
| Safety pass rate | 2/2 (100%) | 100% |
| Data-use correctness | 5/5 (100%) | ≥ 75% |
| Latency | p50 3203 ms, p95 5313 ms | report |

The evaluation run that produced these numbers (public + 22 extra cases) made 44 new paid LLM calls
(tokens in/out 72480/6609, about $0.0148).

### Extra cases (22) — written by us; NOT held-out

These cases were written by the team, and the keyword fallback and several rules were tuned against them while
building (for example the false-availability rule for discrepancies, the POS outage/degraded split, typo and
Hinglish keywords). Treat them as a regression suite, not as evidence of generalisation.

| Metric | Value | Target |
|---|---|---|
| Category accuracy | 22/22 (100%) | ≥ 75% |
| Priority accuracy | 22/22 (100%) | ≥ 75% |
| Escalation accuracy | 22/22 (100%) | report |
| Citation presence | 20/20 (100%) | ≥ 85% |
| Citation validity | 20/20 (100%) | 100% |
| Citation recall | 86% | report |
| Safety pass rate | 3/3 (100%) | 100% |
| Data-use correctness | not measured (no data expectations defined for these cases) | ≥ 75% |
| Latency | p50 5476 ms, p95 6711 ms | report |

### Unseen cases UNS-01..UNS-08 — added by teammates, run before any code change

| Measure | Before any fixes | After fixes |
|---|---|---|
| Category | 8/8 | 8/8 |
| Priority | 7/8 | 7/8 |
| Escalated as expected | 7/8 | 7/8 |
| Safety refusal as expected | 7/8 | 7/8 |
| UNS-08 (vague request) gets a follow-up question | no | yes |
| UNS-08 citations | 4 unrelated retrieved sections | 1 (the rule's own citation) |
| UNS-02 (return with bill and amount) asks for a SKU | yes | no |
| P1 answers (UNS-01, UNS-04) open with safe guidance | no | yes |

The fixes were general behaviour changes, not label or case-specific changes: a follow-up question when confidence
is below 0.60, citation abstention when confidence is low and only the catch-all rule fired, a SKU question only for
categories that need one, and safe guidance first in every P1 answer. They did not change the scored numbers.
The one miss in both runs is **UNS-07**: "paste the CVV and card number…" is not recognised as a payment-data
request, so it is not refused (P3 instead of P1). It is still open.

Cost: the first unseen run made 16 calls (about $0.0050); the reruns made 2 and then 0 new calls (about $0.0004 in
total) thanks to the cache.

## Rules vs LLM-suggested priority

| | Rules (final priority) | LLM-suggested priority |
|---|---|---|
| Public (10 cases that reach triage) | 10/10 | 5/10 |
| Extra (20 cases that reach triage) | 20/20 | 13/20 |
| **Total** | **30/30** | **18/30** |

This is **not a like-for-like model comparison.** The LLM suggests a priority at triage from the masked text only;
the rules decide from the text plus the Data Agent's findings (stock levels, receipts, warehouse stock, complaint
counts, refund thresholds). The gap shows why priority is decided in code: it depends on thresholds and data the
LLM does not see at that point. A full metrics table for `--priority-mode llm` was not measured.

## Design decisions worth defending

- **POS outage vs degraded split.** The documents agree once severity is separated:
  `02_pos_pricing_returns_policy.md#pos-card-reader-runbook` says multiple registers that *"cannot process card
  payments"* are P1; `01_retail_operations_sop.md#priority-rules` lists *"multiple registers degraded"* under P2;
  `escalation_matrix.csv` names the P1 row *"Multiple registers or store-wide payment outage"*. So
  `P1-POS-OUTAGE` fires for outage or unknown severity (fail safe) and `P2-POS-DEGRADED` for slow / intermittent
  but still processing. Team and SLA still come from the matrix's multi-register row.
- **Regex can only escalate severity.** If the regex finds outage words it always wins; regex "degraded" is used
  only when the LLM could not tell. A false P2 on a real outage (a store that cannot take payments waits hours)
  costs more than a false P1 on slow tills (a human downgrades it).
- **Mask before the LLM.** Every input passes `safety.check_input` before any LLM call, cache write or log write;
  card numbers, CVV/OTP/PIN, Aadhaar, PAN, phone and email are masked. Requests *for* restricted or payment data are
  refused and escalated to Security without any LLM call. The output gate re-masks the answer, drops citations
  outside the controller's allowed set and rejects claims that an approval or refund already happened.
- **Two LLM calls per request.** Triage (structured output) and the final response. Priority, escalation,
  thresholds, citations and approval boundaries are deterministic Python (`rules.py`, `controller.py`,
  `safety.py`), each rule tied to a policy section. Blocked requests make zero calls.

## Known limitations

- Regex threat detection can miss novel injection phrasing (mitigated by output gate; the LLM never has raw data access).
- Similar product names may resolve to the wrong SKU → follow-up question when ambiguous.
- Thresholds are configured from policy text; a policy change needs a config update (future: rule registry extracted from KB).
- Synthetic data inconsistencies (regions, employee roles vs case roles) are shown as-is and noted.
- Quality keyword list is heuristic; reported in limitations.
- Regex threat detection can miss novel phrasing: the unseen case UNS-07 ("paste the CVV and card number…") was
  not refused.
- `PRIORITY_MODE=llm` has no full metrics table; only the rules-vs-LLM accuracy row above was measured.
- The Dashboard shows `store_master.region` values as given; some are inconsistent in the source data.
