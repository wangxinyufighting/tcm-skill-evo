"""OpenAI-compatible transport tests with no network calls or real API keys."""
import json
import os
import sys
import warnings
import threading

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tcmskill.config import load_config  # noqa: E402
from tcmskill.llm import DeepSeekLLM, LLMHTTPError, LLMOutputBudgetError  # noqa: E402

openai = pytest.importorskip("openai")
# OpenAI 3.x uses HTTPX2; older supported SDKs use HTTPX.
httpx = pytest.importorskip("httpx2" if hasattr(openai, "DefaultHttpx2Client") else "httpx")


def _completion(fingerprint="test-fingerprint"):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "reported-model",
        "system_fingerprint": fingerprint,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"herbs": ["herb-a"]}',
                "reasoning_content": "test reasoning",
            },
        }],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 10,
            "total_tokens": 30,
            "prompt_cache_hit_tokens": 5,
            "prompt_cache_miss_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 3},
        },
    }


@pytest.fixture
def make_llm(monkeypatch, tmp_path):
    monkeypatch.setenv("TCMSKILL_TEST_API_KEY", "test-key-not-real")
    sdk_factory = openai.OpenAI
    clients = []
    llms = []

    def make(handler):
        def create_sdk(**kwargs):
            client = sdk_factory(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))
            clients.append(client)
            return client

        monkeypatch.setattr(openai, "OpenAI", create_sdk)
        cfg = load_config(os.path.join(ROOT, "configs/default.yaml"), overrides=[
            "llm.base_url=https://test.invalid/v1",
            "llm.model=requested-model",
            "llm.api_key_env=TCMSKILL_TEST_API_KEY",
            f"llm.cache_path={tmp_path / 'cache.sqlite'}",
            "llm.max_retries=1",
            "llm.thinking=false",
            "llm.thinking_max_tokens=16000",
            "llm.json_mode=true",
        ])
        llm = DeepSeekLLM(cfg)
        llms.append(llm)
        assert llm._sdk is not None
        return llm

    yield make
    for client in clients:
        client.close()
    for llm in llms:
        llm.cache.conn.close()


@pytest.mark.parametrize("fingerprint", ["test-fingerprint", [{"version": "default", "start": 0, "end": 213}]])
def test_sdk_preserves_response_without_serializer_warnings(make_llm, fingerprint):
    response = _completion(fingerprint)
    llm = make_llm(lambda request: httpx.Response(200, json=response))
    std, extra = llm._payload([{"role": "user", "content": "Return JSON."}], 64)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        raw = llm._post(std, extra)

    assert raw == response


def test_sdk_chat_json_preserves_content_usage_and_cache(make_llm, tmp_path):
    response = _completion([{"version": "default", "start": 0, "end": 213}])
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json=response)

    llm = make_llm(handle)
    llm.call_log_path = str(tmp_path / "llm_calls.jsonl")
    messages = [{"role": "user", "content": "Return JSON."}]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        first = llm.chat_json(messages, task="prescribe", max_tokens=64)
        cached = llm.chat_json(messages, task="prescribe", max_tokens=64)

    assert first.ok and not first.cached
    assert cached.ok and cached.cached
    for result in (first, cached):
        assert result.data == {"herbs": ["herb-a"]}
        assert result.text == response["choices"][0]["message"]["content"]
        assert result.reasoning == "test reasoning"
        assert result.model == "reported-model"
        assert result.usage == response["usage"]
    assert len(requests) == 1
    assert requests[0]["model"] == "requested-model"
    assert requests[0]["messages"] == messages
    assert requests[0]["max_tokens"] == 64
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert requests[0]["thinking"] == {"type": "disabled"}
    totals = llm.usage.summary()["total"]
    assert (totals["calls"], totals["cached"], totals["errors"]) == (2, 1, 0)
    assert (totals["hit"], totals["miss"], totals["out"], totals["reasoning"]) == (5, 15, 10, 3)
    logs = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert [row["cached"] for row in logs] == [False, True]
    assert all(row["ok"] and row["error"] is None for row in logs)


@pytest.mark.parametrize("status", [400, 401, 429, 500])
def test_sdk_http_errors_keep_status_and_body(make_llm, status):
    body = '{"error": {"message": "test error"}}'
    llm = make_llm(lambda request: httpx.Response(status, text=body, headers={"Content-Type": "application/json"}))
    std, extra = llm._payload([{"role": "user", "content": "Return JSON."}], 64)

    with pytest.raises(LLMHTTPError) as exc:
        llm._post(std, extra)
    assert exc.value.status == status
    assert exc.value.body == body


def test_sdk_parameter_degradation_still_retries(make_llm):
    requests = []
    response = _completion()

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json=response)

    llm = make_llm(handle)
    raw = llm._call([{"role": "user", "content": "Return JSON."}], 64)
    assert raw == response
    assert len(requests) == 2
    assert "response_format" in requests[0]
    assert "response_format" not in requests[1]


def test_exhausted_requests_keep_root_cause_and_do_not_sleep_after_last_attempt(make_llm, monkeypatch):
    requests, sleeps = [], []

    def handle(request):
        requests.append(request)
        raise httpx.ConnectError("test connection reset", request=request) from OSError("test root cause")

    monkeypatch.setattr("tcmskill.llm.time.sleep", sleeps.append)
    llm = make_llm(handle)
    result = llm.chat_json([], "prescribe", meta={"cid": "c1"})
    assert result.request_failed and not result.ok
    assert len(requests) == 2 and len(sleeps) == 1
    assert len(result.attempts) == 1
    attempts = result.attempts[0]["transport_attempts"]
    assert len(attempts) == 2 and all(not a["ok"] for a in attempts)
    assert "retry_delay" in attempts[0] and "retry_delay" not in attempts[1]
    assert any(e["type"] == "ConnectError" for e in result.error_chain)
    assert result.error_chain[-1] == {"type": "OSError", "message": "test root cause"}
    assert llm.cache.get(result.request_key) is None


@pytest.mark.parametrize("failure", ["connection", "http429", "http502", "empty502"])
def test_recovered_transport_retry_is_preserved_in_cache(make_llm, monkeypatch, failure):
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            if failure == "connection":
                raise httpx.ConnectError("test reset", request=request)
            status = 429 if failure == "http429" else 502
            message = "empty reply" if failure == "empty502" else "test temporary failure"
            return httpx.Response(status, json={"error": message})
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr("tcmskill.llm.time.sleep", lambda delay: None)
    llm = make_llm(handle)
    with llm.require_responses():
        result = llm.chat_json([], "reflect")
        cached = llm.chat_json([], "reflect")
    assert result.ok and not result.request_failed and cached.cached
    attempts = result.attempts[0]["transport_attempts"]
    assert [a["ok"] for a in attempts] == [False, True]
    assert cached.attempts == result.attempts
    assert len(requests) == 2
    if failure == "http429":
        assert attempts[0]["http_status"] == 429


@pytest.mark.parametrize("thinking,requested,effective", [(False, 8192, 8192), (True, 8192, 16000), (True, 64000, 64000)])
def test_reasoning_only_502_fails_fast_with_actual_budget_and_diagnostics(make_llm, monkeypatch, tmp_path, thinking, requested, effective):
    requests, sleeps = [], []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(502, json={"error": {"message":
            "WorkBuddy AI: an empty reply (the model answered nothing; try again) - cut at its length, the whole output spent on reasoning"}})
    monkeypatch.setattr("tcmskill.llm.time.sleep", sleeps.append)
    llm = make_llm(handle)
    llm.thinking = thinking
    llm.max_retries = 6
    llm.call_log_path = str(tmp_path / "budget_failure.jsonl")
    with pytest.raises(LLMOutputBudgetError, match=f"max_tokens={effective}"):
        llm.chat_json([{"role": "user", "content": "Return JSON."}], "knowledge_rx", max_tokens=requested)
    assert len(requests) == 1 and sleeps == []
    assert requests[0]["max_tokens"] == effective
    row = json.loads((tmp_path / "budget_failure.jsonl").read_text())
    assert row["request_failed"] and not row["ok"]
    assert len(row["attempts"]) == 1 and row["attempts"][0]["truncated"]
    transport = row["attempts"][0]["transport_attempts"][0]
    assert transport["http_status"] == 502 and transport["max_tokens"] == effective
    assert transport["failure_kind"] == "reasoning_budget_exhausted"
    assert llm.cache.get(row["request_key"]) is None


def test_compatible_thinking_false_does_not_send_a_server_disable_parameter(make_llm):
    llm = make_llm(lambda request: httpx.Response(200, json=_completion()))
    llm.send_thinking = False
    std, extra = llm._payload([], 64000)
    assert std["max_tokens"] == 64000
    assert "thinking" not in extra and "reasoning_effort" not in extra


def test_transport_history_is_thread_local(make_llm, monkeypatch):
    barrier = threading.Barrier(2)
    counts = {}
    lock = threading.Lock()

    def handle(request):
        cid = json.loads(request.content)["messages"][0]["content"]
        with lock:
            counts[cid] = counts.get(cid, 0) + 1
            attempt = counts[cid]
        if attempt == 1:
            barrier.wait(timeout=5)
            raise httpx.ConnectError(f"reset for {cid}", request=request)
        return httpx.Response(200, json=_completion())

    monkeypatch.setattr("tcmskill.llm.time.sleep", lambda delay: None)
    llm = make_llm(handle)
    ids = ["case-one", "case-two"]
    results = llm.map(lambda cid: llm.chat_json([{"role": "user", "content": cid}], "prescribe"), ids, workers=2)
    for cid, result in zip(ids, results):
        assert result.ok
        attempts = result.attempts[0]["transport_attempts"]
        assert [a["ok"] for a in attempts] == [False, True]
        assert cid in attempts[0]["error_chain"][-1]["message"]
