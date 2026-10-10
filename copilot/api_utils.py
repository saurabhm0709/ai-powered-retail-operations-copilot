"""Shared disk cache (.cache/) and paid-call counter (MAX_CALLS) for every OpenAI call (ARCHITECTURE.md §4)."""

import hashlib
import json
from pathlib import Path
from typing import Callable

from copilot import config

# Paid calls made by this process; checked against config.MAX_CALLS.
_call_count = 0


# Inputs that affect the LLM result -> sha256 hex string. Changing any input gives a new key.
def cache_key(masked_text: str, system_prompt: str, model: str, temperature: float,
              max_tokens: int, prompt_version: str, schema_name: str) -> str:
    key_parts = {
        "masked_text": masked_text,
        "system_prompt": system_prompt,
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_version": prompt_version,
        "schema_name": schema_name,
    }
    # sort_keys makes the JSON (and so the hash) independent of dict order
    key_json = json.dumps(key_parts, sort_keys=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()


# Cache key -> path of its JSON file in the cache folder.
def cache_path(key: str) -> Path:
    return Path(config.CACHE_DIR) / f"{key}.json"


# Cache key -> saved result dict, or None if not cached yet.
def read_cache(key: str) -> dict | None:
    path = cache_path(key)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# Saves a result dict under its cache key.
def write_cache(key: str, result: dict) -> None:
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


# Counts one paid call; raises RuntimeError when the MAX_CALLS budget would be exceeded.
def count_call() -> None:
    global _call_count
    if _call_count >= config.MAX_CALLS:
        raise RuntimeError(f"MAX_CALLS budget reached ({config.MAX_CALLS} paid calls); stopping.")
    _call_count += 1


# Number of paid calls made so far (for traces and eval reports).
def call_count() -> int:
    return _call_count


# Resets the paid-call counter (used by tests and at the start of an eval run).
def reset_call_count() -> None:
    global _call_count
    _call_count = 0


# Cache key + function that makes one paid call -> result dict. Cache hit costs nothing.
def cached_call(key: str, call_fn: Callable[[], dict]) -> tuple[dict, bool]:
    cached_result = read_cache(key)
    if cached_result is not None:
        return cached_result, True
    count_call()  # counted before calling, so a failed attempt still uses budget
    result = call_fn()  # exceptions propagate, so failures are never cached
    write_cache(key, result)
    return result, False
