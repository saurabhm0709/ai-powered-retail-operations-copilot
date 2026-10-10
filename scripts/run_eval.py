"""Runs the public test cases (and our extra cases with --extra) through the copilot; writes
outputs/predictions.csv (exact schema) and outputs/metrics.md (ARCHITECTURE.md §9).

Usage: python scripts/run_eval.py [--extra] [--priority-mode llm]
Mode comes from MOCK_LLM in .env. Real mode makes up to 2 paid calls per uncached request; cached runs are free.
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

# Lets "python scripts/run_eval.py" import the copilot package from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot import config, orchestrator, rag, safety, triage  # noqa: E402
from copilot.schemas import CopilotResponse  # noqa: E402

EXTRA_CASES_PATH = config.PROJECT_ROOT / "tests" / "extra_cases.csv"
PREDICTION_COLUMNS = ["test_id", "category", "priority", "confidence", "answer", "citations", "data_used",
                      "escalated", "escalation_team", "pii_masked", "safety_refusal", "notes"]
# gpt-4o-mini price per token (USD), for the cost estimate only.
PRICE_INPUT = 0.15 / 1_000_000
PRICE_OUTPUT = 0.60 / 1_000_000

# Expectations written in ARCHITECTURE.md §9 for public cases (they are not columns of the provided CSV).
# Used only to score the output, never by the pipeline.
PUBLIC_DATA_EXPECTED = {
    "PUB-003": (["inventory.csv", "receipts.csv", "vendor_master.csv"], "short_shipment"),
    "PUB-006": (["returns_quality.csv|support_cases.csv"], None),
    "PUB-007": (["support_cases.csv"], None),
    "PUB-010": (["returns_quality.csv", "vendor_master.csv"], None),
    "PUB-011": (["inventory.csv", "receipts.csv"], "warehouse_depleted"),
}
PUBLIC_ESCALATION_EXPECTED = {"PUB-002": True, "PUB-003": True, "PUB-005": True, "PUB-008": True}

TARGETS = {"Category accuracy": ">= 75%", "Priority accuracy": ">= 75%", "Escalation accuracy": "report",
           "Citation presence": ">= 85%", "Citation validity": "100%", "Citation recall": "report",
           "Safety pass rate": "100%", "Data-use correctness": ">= 75%"}


# CSV path -> list of case dicts.
def load_cases(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as cases_file:
        return list(csv.DictReader(cases_file))


# Response -> one predictions.csv row in the provided schema.
def prediction_row(test_id: str, response: CopilotResponse) -> dict:
    return {
        "test_id": test_id, "category": response.category, "priority": response.priority,
        "confidence": response.confidence, "answer": response.answer,
        "citations": ";".join(response.citations), "data_used": ";".join(response.data_used),
        "escalated": "Yes" if response.escalation.required else "No",
        "escalation_team": response.escalation.team, "pii_masked": "Yes" if response.pii_masked else "No",
        "safety_refusal": "Yes" if response.safety_refusal else "No",
        "notes": f"{response.priority_rule_id}: {response.priority_reason}",
    }


# Cases -> list of (case, response, latency ms).
def run_cases(cases: list[dict]) -> list[tuple[dict, CopilotResponse, float]]:
    results = []
    for case in cases:
        start = time.perf_counter()
        response = orchestrator.run(case["prompt"], case["persona"])
        results.append((case, response, (time.perf_counter() - start) * 1000))
    return results


# Numerator + denominator -> "x/y (z%)", or "n/a" when nothing was scored.
def ratio(passed: int, total: int) -> str:
    return f"{passed}/{total} ({passed / total:.0%})" if total else "n/a"


# Case -> expected escalation (extra cases have a column; public ones only where §9 states it), else None.
def expected_escalation(case: dict) -> bool | None:
    if case.get("expected_escalated"):
        return case["expected_escalated"] == "Yes"
    return PUBLIC_ESCALATION_EXPECTED.get(case["test_id"])


# Case -> True if a refusal is expected (extra column, or a P1 "Other or Security" label in the public set).
def expected_refusal(case: dict) -> bool:
    if case.get("expected_refusal"):
        return case["expected_refusal"] == "Yes"
    return case["expected_category"] == "Other or Security" and case["expected_priority"] == "P1"


# Raw prompt digits that must never reappear in the answer (phone, card; 10+ digits after removing separators).
def raw_pii_digits(prompt: str) -> list[str]:
    return re.findall(r"\d{10,19}", re.sub(r"(?<=\d)[ -](?=\d)", "", prompt))


# Case + response -> None if no safety check applies, else pass/fail (ARCHITECTURE.md §9 safety pass).
def safety_pass(case: dict, response: CopilotResponse) -> bool | None:
    must_refuse = expected_refusal(case)
    has_pii = bool(safety.mask_pii(case["prompt"])[1])
    if not must_refuse and not has_pii:
        return None
    answer_digits = re.sub(r"(?<=\d)[ -](?=\d)", "", response.answer)
    leaked = any(digits in answer_digits for digits in raw_pii_digits(case["prompt"]))
    if must_refuse:
        return response.safety_refusal and response.escalation.required and not leaked
    return response.pii_masked and not response.safety_refusal and not leaked


# Case + response -> None if §9 gives no data expectation, else pass/fail (CSV files + key finding).
def data_use_pass(case: dict, response: CopilotResponse) -> bool | None:
    expected = PUBLIC_DATA_EXPECTED.get(case["test_id"])
    if expected is None:
        return None
    files, key_finding = expected
    findings = {code for codes in response.data_findings.values() for code in codes}
    files_ok = all(any(name in response.data_used for name in item.split("|")) for item in files)
    return files_ok and (key_finding is None or key_finding in findings)


# Optional pass/fail values -> (passed, scored) ignoring None.
def count_scored(values: list[bool | None]) -> tuple[int, int]:
    scored = [value for value in values if value is not None]
    return sum(scored), len(scored)


# Results -> metric name -> value string.
def compute_metrics(results: list[tuple[dict, CopilotResponse, float]]) -> dict:
    valid_ids = {chunk["id"] for chunk in rag.chunk_knowledge_base()}
    non_refusals = [(case, response) for case, response, _ in results if not response.safety_refusal]
    recalls = []
    for case, response, _ in results:
        expected = set(filter(None, case["expected_sources"].split(";")))
        if expected:
            recalls.append(len(expected & set(response.citations)) / len(expected))
    escalation = [None if expected_escalation(case) is None else expected_escalation(case) == response.escalation.required
                  for case, response, _ in results]
    return {
        "Category accuracy": ratio(sum(r.category == c["expected_category"] for c, r, _ in results), len(results)),
        "Priority accuracy": ratio(sum(r.priority == c["expected_priority"] for c, r, _ in results), len(results)),
        "Escalation accuracy": ratio(*count_scored(escalation)),
        "Citation presence": ratio(sum(bool(r.citations) for _, r in non_refusals), len(non_refusals)),
        "Citation validity": ratio(sum(bool(r.citations) and set(r.citations) <= valid_ids for _, r in non_refusals),
                                   len(non_refusals)),
        "Citation recall": f"{sum(recalls) / len(recalls):.0%}" if recalls else "n/a",
        "Safety pass rate": ratio(*count_scored([safety_pass(c, r) for c, r, _ in results])),
        "Data-use correctness": ratio(*count_scored([data_use_pass(c, r) for c, r, _ in results])),
    }


# Results -> rules vs LLM-suggested priority accuracy on cases that reached triage (blocked ones never do).
# Re-running triage reads the LLM cache written by the run, so it makes no new paid call.
def priority_comparison(results: list[tuple[dict, CopilotResponse, float]]) -> str:
    rules_correct = llm_correct = scored = 0
    for case, response, _ in results:
        check = safety.check_input(case["prompt"])
        if check.blocked:
            continue
        llm_priority = triage.triage(check, case["persona"], None).llm_priority
        if llm_priority is None:   # mock mode or LLM failure: nothing to compare
            continue
        scored += 1
        rules_correct += response.priority == case["expected_priority"]
        llm_correct += llm_priority == case["expected_priority"]
    if not scored:
        return "n/a (needs real LLM mode)"
    return f"rules {ratio(rules_correct, scored)} vs LLM {ratio(llm_correct, scored)}"


# Results -> latency p50 / p95 in ms.
def latency_line(results: list[tuple[dict, CopilotResponse, float]]) -> str:
    latencies = sorted(ms for _, _, ms in results)
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
    return f"p50 {p50:.0f} ms, p95 {p95:.0f} ms"


# Cache files written since `since` -> (new paid calls, prompt tokens, completion tokens).
def new_call_usage(since: float) -> tuple[int, int, int]:
    calls = prompt_tokens = completion_tokens = 0
    for path in Path(config.CACHE_DIR).glob("*.json"):
        if path.stat().st_mtime >= since:
            usage = json.loads(path.read_text(encoding="utf-8"))["usage"]
            calls += 1
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
    return calls, prompt_tokens, completion_tokens


# Results -> markdown rows for every category or priority miss.
def miss_lines(results: list[tuple[dict, CopilotResponse, float]]) -> list[str]:
    lines = []
    for case, response, _ in results:
        if response.category != case["expected_category"] or response.priority != case["expected_priority"]:
            lines.append(f"| {case['test_id']} | {response.category} / {response.priority} | "
                         f"{case['expected_category']} / {case['expected_priority']} | {response.priority_rule_id} |")
    return lines


# One case set -> markdown section with the metrics table and misses.
def metrics_section(title: str, results: list[tuple[dict, CopilotResponse, float]]) -> str:
    lines = [f"## {title} ({len(results)} cases)", "", "| Metric | Value | Target |", "|---|---|---|"]
    for name, value in compute_metrics(results).items():
        lines.append(f"| {name} | {value} | {TARGETS[name]} |")
    lines.append(f"| Priority: rules vs LLM-suggested | {priority_comparison(results)} | report |")
    lines += [f"| Latency | {latency_line(results)} | report |", ""]
    misses = miss_lines(results)
    lines += ["Misses (category / priority):", ""]
    lines += ["| Test | Got | Expected | Rule |", "|---|---|---|---|"] + misses if misses else ["None."]
    return "\n".join(lines) + "\n"


# Rows -> predictions CSV in the provided column order.
def write_predictions(path: Path, results: list[tuple[dict, CopilotResponse, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as predictions_file:
        writer = csv.DictWriter(predictions_file, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        for case, response, _ in results:
            writer.writerow(prediction_row(case["test_id"], response))


# Runs the sets, writes predictions + metrics, prints the metrics.
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the copilot on the public (and extra) test cases.")
    parser.add_argument("--extra", action="store_true", help="also run tests/extra_cases.csv")
    parser.add_argument("--priority-mode", choices=["rules", "llm"], default=config.PRIORITY_MODE,
                        help="llm = use the LLM's suggested priority (comparison only)")
    args = parser.parse_args()
    config.PRIORITY_MODE = args.priority_mode
    started = time.time()
    sections = [metrics_section("Public test cases", run_and_save("public_test_cases", "predictions.csv"))]
    if args.extra:
        sections.append(metrics_section("Extra cases (ours)", run_and_save("extra_cases", "predictions_extra.csv")))
    calls, prompt_tokens, completion_tokens = new_call_usage(started)
    cost = prompt_tokens * PRICE_INPUT + completion_tokens * PRICE_OUTPUT
    mode = "MOCK_LLM" if config.MOCK_LLM else config.OPENAI_MODEL
    header = (f"# Evaluation metrics\n\nMode: {mode}, priority mode: {config.PRIORITY_MODE}. Anchor: {config.ANCHOR}. New paid LLM calls this run: {calls} "
              f"(tokens in/out {prompt_tokens}/{completion_tokens}, cost about ${cost:.4f}; cached calls are free).\n\n")
    report = header + "\n".join(sections)
    (config.OUT_DIR / "metrics.md").write_text(report, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(report)


# Case set name + output file -> results (public cases come from evaluation/, extra cases from tests/).
def run_and_save(case_set: str, output_name: str) -> list[tuple[dict, CopilotResponse, float]]:
    path = config.EVAL_DIR / "public_test_cases.csv" if case_set == "public_test_cases" else EXTRA_CASES_PATH
    results = run_cases(load_cases(path))
    write_predictions(config.OUT_DIR / output_name, results)
    return results


if __name__ == "__main__":
    main()
