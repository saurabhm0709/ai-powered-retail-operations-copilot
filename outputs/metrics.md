# Evaluation metrics

Mode: gpt-4o-mini. Anchor: 2026-10-10 12:00:00. New paid LLM calls this run: 24 (tokens in/out 34776/4112, cost about $0.0077; cached calls are free).

## Public test cases (11 cases)

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
| Latency | p50 3482 ms, p95 4429 ms | report |

Misses (category / priority):

None.

## Extra cases (ours) (8 cases)

| Metric | Value | Target |
|---|---|---|
| Category accuracy | 8/8 (100%) | >= 75% |
| Priority accuracy | 8/8 (100%) | >= 75% |
| Escalation accuracy | 8/8 (100%) | report |
| Citation presence | 7/7 (100%) | >= 85% |
| Citation validity | 7/7 (100%) | 100% |
| Citation recall | 88% | report |
| Safety pass rate | 1/1 (100%) | 100% |
| Data-use correctness | n/a | >= 75% |
| Latency | p50 5396 ms, p95 7963 ms | report |

Misses (category / priority):

None.
