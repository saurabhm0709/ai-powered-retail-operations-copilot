# Evaluation metrics

Mode: gpt-4o-mini, priority mode: rules. Anchor: 2026-10-10 12:00:00. New paid LLM calls this run: 44 (tokens in/out 72480/6609, cost about $0.0148; cached calls are free).

## Public test cases (11 cases)

Note on "Priority: rules vs LLM-suggested": the LLM suggests a priority from the masked text only, at triage; the rules decide from the text plus the Data Agent's findings.

| Metric | Value | Target |
|---|---|---|
| Category accuracy | 11/11 (100%) | >= 75% |
| Priority accuracy | 11/11 (100%) | >= 75% |
| Escalation accuracy | 4/4 (100%) | report |
| Citation presence | 10/10 (100%) | >= 85% |
| Citation validity | 10/10 (100%) | 100% |
| Citation recall | 100% | report |
| Safety pass rate | 2/2 (100%) | 100% |
| Data-use correctness | 5/5 (100%) | >= 75% |
| Priority: rules vs LLM-suggested | rules 10/10 (100%) vs LLM 5/10 (50%) | report |
| Latency | p50 3203 ms, p95 5313 ms | report |

Misses (category / priority):

None.

## Extra cases (ours) (22 cases)

Note on "Priority: rules vs LLM-suggested": the LLM suggests a priority from the masked text only, at triage; the rules decide from the text plus the Data Agent's findings.

| Metric | Value | Target |
|---|---|---|
| Category accuracy | 22/22 (100%) | >= 75% |
| Priority accuracy | 22/22 (100%) | >= 75% |
| Escalation accuracy | 22/22 (100%) | report |
| Citation presence | 20/20 (100%) | >= 85% |
| Citation validity | 20/20 (100%) | 100% |
| Citation recall | 86% | report |
| Safety pass rate | 3/3 (100%) | 100% |
| Data-use correctness | n/a | >= 75% |
| Priority: rules vs LLM-suggested | rules 20/20 (100%) vs LLM 13/20 (65%) | report |
| Latency | p50 5476 ms, p95 6711 ms | report |

Misses (category / priority):

None.
