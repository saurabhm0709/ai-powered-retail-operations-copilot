"""OpenAI structured-output call: cache -> counter -> call -> retry(2), with a mock switch (ARCHITECTURE.md §4)."""

import logging
import os
import time
from functools import lru_cache

import openai
from pydantic import BaseModel

from copilot import api_utils, config

logger = logging.getLogger(__name__)

# Errors a retry cannot fix (wrong key, no access to the model): stop after the first attempt.
NON_RETRYABLE_ERRORS = (openai.AuthenticationError, openai.PermissionDeniedError)


# Reason -> one warning line. Only the error class name is logged: auth error messages can contain part of the key.
def warn_fallback(reason: str) -> None:
    logger.warning("LLM unavailable (%s); using the deterministic fallback.", reason)


# Creates the OpenAI client once; it reads OPENAI_API_KEY from the environment (.env).
@lru_cache(maxsize=1)
def get_client() -> openai.OpenAI:
    return openai.OpenAI()


# Usage dict with zero tokens, for mock mode and total failure.
def empty_usage(mock: bool = False) -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "cached": False, "mock": mock}


# System + user prompt + schema -> one paid OpenAI call -> {"parsed": ..., "usage": ...}. Raises on failure.
def call_openai(system: str, user: str, schema: type[BaseModel], max_tokens: int) -> dict:
    completion = get_client().chat.completions.parse(
        model=config.OPENAI_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format=schema,
        temperature=config.TEMPERATURE,
        max_tokens=max_tokens,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:  # model refused or returned no valid JSON
        raise ValueError("LLM returned no parsed output")
    usage = {"prompt_tokens": completion.usage.prompt_tokens,
             "completion_tokens": completion.usage.completion_tokens}
    return {"parsed": parsed.model_dump(), "usage": usage}


# Masked user text + system prompt + Pydantic schema -> (schema instance | None, usage).
# None means the caller must use its deterministic fallback. Only masked text may be passed in.
def structured(system: str, user: str, schema: type[BaseModel],
               max_tokens: int) -> tuple[BaseModel | None, dict]:
    if config.MOCK_LLM:
        return None, empty_usage(mock=True)
    if not os.getenv("OPENAI_API_KEY"):  # no key: no call is made or counted
        warn_fallback("OPENAI_API_KEY is not set")
        return None, empty_usage()
    key = api_utils.cache_key(user, system, config.OPENAI_MODEL, config.TEMPERATURE,
                              max_tokens, config.PROMPT_VERSION, schema.__name__)
    for attempt in range(config.MAX_RETRIES + 1):
        try:
            result, from_cache = api_utils.cached_call(
                key, lambda: call_openai(system, user, schema, max_tokens))
            usage = {**result["usage"], "cached": from_cache, "mock": False}
            return schema.model_validate(result["parsed"]), usage
        except NON_RETRYABLE_ERRORS as error:
            warn_fallback(type(error).__name__)
            return None, empty_usage()
        except (openai.OpenAIError, ValueError):  # API failure or invalid JSON; budget RuntimeError is not caught
            if attempt < config.MAX_RETRIES:
                time.sleep(config.RETRY_BACKOFF_SECONDS * (2 ** attempt))
    warn_fallback(f"failed after {config.MAX_RETRIES + 1} attempts")
    return None, empty_usage()
