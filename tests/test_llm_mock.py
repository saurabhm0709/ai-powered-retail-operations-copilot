"""Tests for api_utils.py and llm.py. No network: the OpenAI client is always replaced by a fake."""

from types import SimpleNamespace

import httpx
import openai
import pytest
from pydantic import BaseModel

from copilot import api_utils, config, llm


class Answer(BaseModel):
    text: str


# Fake OpenAI client: returns `parsed` from every parse() call and counts the calls.
def make_fake_client(parsed: BaseModel | None) -> SimpleNamespace:
    calls = []

    def parse(**kwargs):
        calls.append(kwargs)
        message = SimpleNamespace(parsed=parsed)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)

    completions = SimpleNamespace(parse=parse)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions), calls=calls)


# Every test: temp cache folder, fresh counter, no sleeping, real mode, and no real client.
@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(config, "MOCK_LLM", False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setattr(llm.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(llm, "get_client", lambda: pytest.fail("real OpenAI client used"))
    api_utils.reset_call_count()


def test_mock_mode_returns_none_without_client(monkeypatch):
    monkeypatch.setattr(config, "MOCK_LLM", True)
    result, usage = llm.structured("sys", "user text", Answer, max_tokens=50)
    assert result is None
    assert usage["mock"] is True
    assert api_utils.call_count() == 0


def test_cache_key_changes_with_every_input():
    base = ("masked", "sys", "gpt-4o-mini", 0, 400, "v1", "Answer")
    base_key = api_utils.cache_key(*base)
    assert api_utils.cache_key(*base) == base_key
    for position in range(len(base)):
        changed = list(base)
        changed[position] = "different"
        assert api_utils.cache_key(*changed) != base_key


def test_second_identical_call_uses_cache(monkeypatch):
    fake = make_fake_client(Answer(text="hello"))
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    first, first_usage = llm.structured("sys", "user text", Answer, max_tokens=50)
    second, second_usage = llm.structured("sys", "user text", Answer, max_tokens=50)
    assert first.text == "hello" and second.text == "hello"
    assert first_usage["cached"] is False and second_usage["cached"] is True
    assert len(fake.calls) == 1
    assert api_utils.call_count() == 1


def test_failure_retries_twice_then_returns_none(monkeypatch):
    fake = make_fake_client(None)  # parsed=None simulates invalid / refused output
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    result, usage = llm.structured("sys", "user text", Answer, max_tokens=50)
    assert result is None
    assert len(fake.calls) == config.MAX_RETRIES + 1
    assert list(config.CACHE_DIR.iterdir()) == []  # failures are never cached


def test_call_counter_raises_past_max_calls(monkeypatch):
    monkeypatch.setattr(config, "MAX_CALLS", 2)
    api_utils.count_call()
    api_utils.count_call()
    with pytest.raises(RuntimeError):
        api_utils.count_call()


def test_budget_error_is_not_swallowed_by_retries(monkeypatch):
    monkeypatch.setattr(config, "MAX_CALLS", 0)
    monkeypatch.setattr(llm, "get_client", lambda: make_fake_client(Answer(text="x")))
    with pytest.raises(RuntimeError):
        llm.structured("sys", "user text", Answer, max_tokens=50)


def test_missing_key_skips_call_without_counting(monkeypatch, caplog):
    monkeypatch.delenv("OPENAI_API_KEY")
    result, usage = llm.structured("sys", "user text", Answer, max_tokens=50)
    assert result is None
    assert api_utils.call_count() == 0
    assert "OPENAI_API_KEY is not set" in caplog.text


def test_auth_error_is_not_retried_and_key_text_not_logged(monkeypatch, caplog):
    response = httpx.Response(401, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
    error = openai.AuthenticationError("Incorrect API key provided: sk-abc...wxyz", response=response, body=None)
    fake = make_fake_client(None)
    attempts = []

    def failing_parse(**kwargs):
        attempts.append(kwargs)
        raise error

    fake.chat.completions.parse = failing_parse
    monkeypatch.setattr(llm, "get_client", lambda: fake)
    result, usage = llm.structured("sys", "user text", Answer, max_tokens=50)
    assert result is None
    assert len(attempts) == 1
    assert "AuthenticationError" in caplog.text
    assert "sk-abc" not in caplog.text
