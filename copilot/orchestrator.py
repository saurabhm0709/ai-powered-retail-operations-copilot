"""Runs the pipeline stages in order and writes one masked trace per request (ARCHITECTURE.md §7)."""

import json
import time
import uuid
from datetime import datetime

from copilot import api_utils, config, controller, data_agent, rag, responder, safety, triage
from copilot.schemas import CopilotResponse


# New trace dict: id, start time, per-agent timings and the paid-call count at the start.
def new_trace(persona: str | None) -> dict:
    return {"trace_id": uuid.uuid4().hex[:12], "timestamp": datetime.now().isoformat(timespec="seconds"),
            "persona": persona, "agent_ms": {}, "calls_at_start": api_utils.call_count(), "data_tools": []}


# Runs one stage function, records its duration in the trace, returns its result.
def timed(trace: dict, stage: str, stage_fn, *args):
    start = time.perf_counter()
    result = stage_fn(*args)
    trace["agent_ms"][stage] = round((time.perf_counter() - start) * 1000, 1)
    return result


# Appends one JSON line to traces.jsonl. Only masked text is ever written (AGENTS.md rule 2).
def write_trace(trace: dict) -> None:
    config.TRACES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.TRACES_PATH, "a", encoding="utf-8") as trace_file:
        trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")


# Final response + trace -> response with trace_id set; trace record completed and saved.
def finish(response: CopilotResponse, trace: dict) -> CopilotResponse:
    trace["llm_calls"] = api_utils.call_count() - trace.pop("calls_at_start")
    trace["model"] = "mock" if config.MOCK_LLM else config.OPENAI_MODEL
    trace.update(rule_id=response.priority_rule_id, priority=response.priority, citations=response.citations)
    write_trace(trace)
    return response.model_copy(update={"trace_id": trace["trace_id"]})


# User query (+ optional persona and store from the UI) -> CopilotResponse.
def run(query: str, persona: str | None = None, store_hint: str | None = None) -> CopilotResponse:
    trace = new_trace(persona)
    safety_check = timed(trace, "safety", safety.check_input, query)            # [4a] mask before anything else
    trace["masked_query"] = safety_check.masked_text
    if safety_check.blocked:
        return finish(controller.refusal(safety_check), trace)
    triage_result = timed(trace, "triage", triage.triage, safety_check, persona, store_hint)   # [2]
    hits = []
    if triage_result.needs_policy:
        hits = timed(trace, "rag", rag.retrieve, safety_check.masked_text, triage_result.category)  # [3A]
    data_results = []
    if triage_result.needs_data:
        data_results = timed(trace, "data", data_agent.run_data_agent, triage_result)          # [3B]
    trace["data_tools"] = [result.tool for result in data_results]
    decision = timed(trace, "controller", controller.decide, safety_check, triage_result, hits, data_results)
    response = timed(trace, "responder", responder.compose,
                     safety_check, triage_result, hits, data_results, decision)              # [5]
    return finish(response, trace)
