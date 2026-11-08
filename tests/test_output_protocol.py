"""Output-contract regressions: scripted responses only, no provider calls."""
import copy
import json
import os
import sys
import runpy
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tcmskill.config import load_config  # noqa: E402
from tcmskill.herbs import HerbLexicon, IncompatibilityChecker  # noqa: E402
from tcmskill.llm import DeepSeekLLM, FatalLLMError, LLMRequestError, parse_json_with_status  # noqa: E402
from tcmskill.prescriber import Prediction, Prescriber  # noqa: E402
from tcmskill.runner import check_stream_resume, evaluate_run, format_output_stats, run_stream, summarize_outputs  # noqa: E402


def response(text='{"herbs":["甘草"]}', finish="stop", model="test-model", reasoning="", tokens=12):
    return {"model": model, "choices": [{"finish_reason": finish, "message": {
        "content": text, "reasoning_content": reasoning,
    }}], "usage": {"prompt_tokens": 20, "completion_tokens": tokens,
                   "completion_tokens_details": {"reasoning_tokens": 5}}}


class ScriptedLLM(DeepSeekLLM):
    def _call(self, messages, max_tokens):
        self.requests.append(self._payload(messages, max_tokens))
        if not self.responses:
            raise AssertionError("Unexpected provider call")
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return copy.deepcopy(value)


@pytest.fixture
def make_llm(tmp_path, monkeypatch):
    monkeypatch.setenv("TCMSKILL_TEST_API_KEY", "not-a-real-key")
    instances = []

    def make(responses, method="ours", overrides=()):
        cfg = load_config(os.path.join(ROOT, "configs/default.yaml"), method=method, overrides=[
            "llm.base_url=https://test.invalid/v1", "llm.model=test-model",
            "llm.api_key_env=TCMSKILL_TEST_API_KEY", "llm.thinking=false",
            "llm.max_tokens=4096", "llm.json_repair_retries=0", "llm.accept_repaired_json=false",
            f"llm.cache_path={tmp_path / f'cache-{len(instances)}.sqlite'}", *overrides,
        ])
        llm = ScriptedLLM(cfg)
        llm.responses, llm.requests = list(responses), []
        llm.call_log_path = str(tmp_path / f"calls-{len(instances)}.jsonl")
        instances.append(llm)
        return llm

    yield make
    for llm in instances:
        llm.cache.conn.close()
        if llm._sdk is not None:
            llm._sdk.close()


@pytest.fixture
def resources():
    lex = HerbLexicon(["甘草", "茯苓"], os.path.join(ROOT, "kb/herb_aliases.json"))
    case = SimpleNamespace(cid="c1", text="咳嗽", symptoms=["咳嗽"], herbs=["甘草"], flags={})
    return SimpleNamespace(
        labels=["甘草", "茯苓"], herb_freq={"甘草": 3, "茯苓": 2}, median_len=1,
        text=lambda c: c.text, lexicon=lex, herb_info=None, cases={"c1": case},
        incompat=IncompatibilityChecker(os.path.join(ROOT, "kb/incompatibility.json"), lex),
    )


@pytest.mark.parametrize("text,status", [
    ('{"herbs":[]}', "complete"), ('```json\n{"herbs":[]}\n```', "extracted"),
    ('Answer: {"herbs":[]}', "extracted"), ("", "empty"), ("null", "invalid"), ("42", "invalid"),
])
def test_json_status(text, status):
    assert parse_json_with_status(text)[1] == status


@pytest.mark.parametrize("text", ['{"herbs":["甘草"]', '[{"herbs":["甘草"]}'])
def test_incomplete_outer_container_is_not_extracted_as_complete(text):
    pytest.importorskip("json_repair")
    assert parse_json_with_status(text)[1] == "repaired"


@pytest.mark.parametrize("method", ["zeroshot", "zeroshot_cot", "case_rag", "ours"])
@pytest.mark.parametrize("thinking,budget", [(False, 4096), (True, 16000)])
def test_baselines_share_output_budget(make_llm, resources, method, thinking, budget):
    llm = make_llm([response()], method=method, overrides=[f"llm.thinking={str(thinking).lower()}"])
    prescriber = Prescriber(llm, resources, llm.cfg)
    pred = prescriber.prescribe(resources.cases["c1"], {})
    assert pred.ok and pred.max_tokens == budget
    assert llm.requests[0][0]["max_tokens"] == budget
    assert prescriber.style == ("direct" if method == "zeroshot" else "structured")


def test_repaired_fragment_is_rejected_not_scored(make_llm, resources):
    pytest.importorskip("json_repair")
    llm = make_llm([response('{"herbs":["甘草"]', "length")])
    pred = Prescriber(llm, resources, llm.cfg).prescribe(resources.cases["c1"], {})
    assert pred.json_status == "repaired" and pred.finish_reason == "length"
    assert not pred.ok and pred.truncated and pred.error == "truncated_output"
    assert pred.herbs == [] and pred.rejected_herbs == ["甘草"]
    assert pred.entries == [] and pred.used_skills == []
    assert pred.empty_prescription is False
    assert llm.cache.get(pred.request_key) is None


@pytest.mark.parametrize("accept", [False, True])
def test_complete_json_with_length_finish_is_always_rejected(make_llm, accept):
    llm = make_llm([response(finish="length")], overrides=[f"llm.accept_repaired_json={str(accept).lower()}"])
    result = llm.chat_json([], "prescribe")
    assert result.json_status == "complete" and not result.ok
    assert result.error == "truncated_output"


def test_syntax_repair_requires_explicit_opt_in(make_llm):
    pytest.importorskip("json_repair")
    for accept in (False, True):
        llm = make_llm([response('{"herbs":["甘草"]')], overrides=[f"llm.accept_repaired_json={str(accept).lower()}"])
        result = llm.chat_json([], "prescribe")
        assert result.json_status == "repaired" and result.ok == accept


def test_retry_and_cache_preserve_both_termination_reasons(make_llm, resources):
    llm = make_llm([response("", "length", reasoning="first", tokens=4096),
                    response(reasoning="second", model="second-model", tokens=12)],
                   overrides=["llm.json_repair_retries=1"])
    prescriber = Prescriber(llm, resources, llm.cfg)
    first = prescriber.prescribe(resources.cases["c1"], {})
    cached = prescriber.prescribe(resources.cases["c1"], {})
    assert first.ok and cached.ok and cached.cached
    assert first.attempts == cached.attempts
    assert [a["finish_reason"] for a in first.attempts] == ["length", "stop"]
    assert [a["ok"] for a in first.attempts] == [False, True]
    assert first.finish_reason == "stop" and not first.truncated
    assert len(llm.requests) == 2
    assert all(request[0]["max_tokens"] == 4096 for request in llm.requests)
    hit = llm.cache.get(first.request_key)["result"]
    assert hit["reasoning"] == "second" and hit["model"] == "second-model"
    total = llm.usage.summary()["total"]
    assert (total["calls"], total["cached"], total["errors"], total["out"]) == (3, 1, 1, 4108)
    output = llm.usage.summary()["output_by_task"]["prescribe"]
    assert (output["requests"], output["cached"], output["accepted_json"]) == (2, 1, 2)
    assert output["finish_reasons"] == {"stop": 2} and output["first_finish_reasons"] == {"length": 2}
    assert output["had_truncation"] == 2
    with open(llm.call_log_path, encoding="utf-8") as f:
        logs = [json.loads(line) for line in f]
    assert [r["cached"] for r in logs] == [False, True]
    assert all(r["cid"] == "c1" and r["request_key"] == first.request_key for r in logs)
    assert logs[0]["attempts"][0]["finish_reason"] == "length"
    stats = summarize_outputs([first])
    assert stats["counts"]["json_complete"] == stats["counts"]["had_truncation"] == 1
    assert stats["counts"]["truncated"] == 0 and stats["counts"]["json_retried"] == 1


def test_cache_key_includes_effective_budget_endpoint_and_policy(make_llm):
    llm = make_llm([], overrides=["llm.thinking=true"])
    original = llm._key([], 1200)
    llm.thinking_max_tokens = 32000
    assert llm._key([], 1200) != original
    llm.thinking_max_tokens = 16000
    llm.base_url += "/another"
    assert llm._key([], 1200) != original
    llm.base_url = llm.cfg.llm.base_url
    llm.accept_repaired_json = True
    assert llm._key([], 1200) != original


def test_legacy_cache_entry_without_diagnostics_is_not_reused(make_llm):
    llm = make_llm([response()])
    llm.cache.set(llm._key([], 4096), {"text": '{"herbs":[]}', "model": "old-model"})
    result = llm.chat_json([], "prescribe")
    assert result.ok and not result.cached and result.finish_reason == "stop"
    assert len(llm.requests) == 1


def test_unknown_finish_reason_is_not_invented(make_llm):
    raw = response()
    del raw["choices"][0]["finish_reason"]
    result = make_llm([raw]).chat_json([], "prescribe")
    assert result.ok and result.finish_reason is None
    assert result.attempts[0]["finish_reason"] is None


def test_retry_transport_error_does_not_reuse_previous_usage(make_llm):
    llm = make_llm([response("", "length", tokens=4096), RuntimeError("transport failed")],
                   overrides=["llm.json_repair_retries=1"])
    result = llm.chat_json([], "prescribe")
    assert not result.ok and result.usage == {} and result.finish_reason is None
    assert len(result.attempts) == 2 and llm.usage.summary()["total"]["out"] == 4096


def test_regeneration_keeps_original_prompt_without_fabricating_assistant_history(make_llm):
    llm = make_llm([response("", "length"), response()], overrides=["llm.json_repair_retries=1"])
    messages = [{"role": "system", "content": "Return a prescription."}, {"role": "user", "content": "case-c1"}]
    original = copy.deepcopy(messages)
    assert llm.chat_json(messages, "prescribe").ok
    assert messages == original
    retry = llm.requests[1][0]["messages"]
    assert [m["role"] for m in retry] == ["system", "user"]
    assert retry[-1]["content"].startswith("case-c1")


def test_fatal_error_is_logged_and_still_raised(make_llm):
    llm = make_llm([FatalLLMError("unauthorized")])
    with pytest.raises(FatalLLMError):
        llm.chat_json([], "prescribe", meta={"cid": "c1"})
    with open(llm.call_log_path, encoding="utf-8") as f:
        row = json.loads(f.readline())
    assert not row["ok"] and row["cid"] == "c1"
    assert row["finish_reason"] is None and "unauthorized" in row["error"]


def test_empty_prescription_and_json_syntax_are_independent(make_llm, resources, tmp_path):
    llm = make_llm([response('{"herbs":[]}')])
    pred = Prescriber(llm, resources, llm.cfg).prescribe(resources.cases["c1"], {})
    assert not pred.ok and pred.error == "empty_prescription"
    assert pred.json_status == "complete" and pred.empty_prescription is True
    agg = evaluate_run(resources, [pred], str(tmp_path))
    assert agg["overall"]["f1"] == 0
    assert agg["overall"]["json_complete"] == agg["overall"]["empty_prescription"] == 1
    stats = json.loads((tmp_path / "test_output_stats.json").read_text())
    assert stats["counts"]["empty_prescription"] == 1 and stats["finish_reasons"] == {"stop": 1}
    assert "空处方" in format_output_stats(stats)


def test_empty_response_is_not_mislabeled_as_empty_prescription(make_llm, resources):
    llm = make_llm([response("")])
    pred = Prescriber(llm, resources, llm.cfg).prescribe(resources.cases["c1"], {})
    assert not pred.ok and pred.json_status == "empty" and pred.empty_prescription is None
    stats = summarize_outputs([pred])
    assert stats["counts"]["json_empty"] == 1 and stats["counts"]["empty_prescription"] == 0


def test_old_predictions_remain_unknown():
    pred = Prediction.from_dict({"cid": "legacy", "herbs": ["甘草"], "ok": True})
    stats = summarize_outputs([pred])
    assert stats["counts"]["json_unknown"] == 1 and stats["empty_prescription_checked"] == 0
    assert stats["finish_reasons"] == {"unknown": 1}


def test_invalid_and_empty_responses_remain_separate(make_llm):
    empty = make_llm([response("")]).chat_json([], "reflect")
    invalid = make_llm([response("42")]).chat_json([], "reflect")
    assert not empty.ok and empty.json_status == "empty"
    assert not invalid.ok and invalid.json_status == "invalid"


def test_request_failure_has_separate_diagnostics_and_stays_in_denominator(make_llm, resources, tmp_path):
    llm = make_llm([RuntimeError("connection failed")])
    pred = Prescriber(llm, resources, llm.cfg).prescribe(resources.cases["c1"], {})
    assert not pred.ok and pred.request_failed
    assert pred.json_status == "unknown" and pred.empty_prescription is None
    assert pred.herbs == [] and pred.finish_reason is None
    stats = summarize_outputs([pred])
    assert stats["counts"]["request_failed"] == 1
    assert stats["counts"]["json_unknown"] == stats["counts"]["empty_prescription"] == 0
    assert "请求失败 | 1 |" in format_output_stats(stats)
    metrics = evaluate_run(resources, [pred], str(tmp_path))
    assert metrics["overall"]["n"] == 1 and metrics["overall"]["f1"] == 0
    assert metrics["overall"]["request_failed"] == 1
    output = llm.usage.summary()["output_by_task"]["prescribe"]
    assert output["request_failures"] == 1
    assert llm.cache.get(pred.request_key) is None


def test_legacy_output_stats_do_not_invent_zero_request_failures():
    stats = summarize_outputs([Prediction(cid="old")])
    del stats["counts"]["request_failed"]
    del stats["rates"]["request_failed"]
    assert "请求失败 | 未知 | 未知 |" in format_output_stats(stats)


@pytest.mark.parametrize("task", ["prescribe", "reflect", "induce", "split"])
def test_learning_requires_responses_but_evaluation_can_record_failures(make_llm, task):
    llm = make_llm([RuntimeError("connection failed"), RuntimeError("connection failed")])
    with pytest.raises(LLMRequestError):
        with llm.require_responses():
            llm.chat_json([], task, meta={"cid": "c1"})
    assert not llm._require_responses
    result = llm.chat_json([], task)
    assert not result.ok and result.request_failed
    with open(llm.call_log_path, encoding="utf-8") as f:
        rows = [json.loads(s) for s in f]
    assert len(rows) == 2 and all(x["request_failed"] for x in rows)
    assert rows[0]["error_chain"][0]["type"] == "RuntimeError"


def test_response_guard_does_not_turn_model_errors_into_connection_errors(make_llm):
    llm = make_llm([response("42")])
    with llm.require_responses():
        result = llm.chat_json([], "reflect")
    assert not result.ok and not result.request_failed
    assert result.error == "invalid_json"


def test_existing_success_cache_without_new_fields_is_compatible(make_llm):
    llm = make_llm([response()])
    first = llm.chat_json([], "prescribe")
    cached = llm.cache.get(first.request_key)
    cached["result"].pop("request_failed")
    cached["result"].pop("error_chain")
    llm.cache.set(first.request_key, cached)
    with llm.require_responses():
        result = llm.chat_json([], "prescribe")
    assert result.cached and result.ok and not result.request_failed
    assert len(llm.requests) == 1


class WindowMethod:
    needs_attempt = True
    learns = True

    def __init__(self, llm):
        self.llm = llm
        self.step = self.window = 0
        self.seen = []

    def warmup(self, train_ids):
        pass

    def call(self, case, task):
        return self.llm.chat_json([{"role": "user", "content": f"{case.cid} {task}"}], task, meta={"cid": case.cid})

    def predict(self, cases, frozen=False):
        preds = []
        for case in cases:
            result = self.call(case, "prescribe")
            preds.append(Prediction(cid=case.cid, herbs=["herb-a"], ok=result.ok,
                                    request_failed=result.request_failed))
        return preds

    def observe(self, cases, preds):
        self.seen.extend(c.cid for c in cases)
        for task in ("reflect", "induce"):
            for case in cases:
                self.call(case, task)
        self.step += len(cases)
        self.window += 1
        return {}

    def summary(self):
        return {"seen": len(self.seen)}

    def state_dict(self):
        return {"step": self.step, "window": self.window, "seen": self.seen}

    def load_state_dict(self, state):
        self.step, self.window = state["step"], state["window"]
        self.seen = list(state["seen"])


@pytest.mark.parametrize("failed_task", ["prescribe", "reflect", "induce"])
def test_failed_window_is_not_checkpointed_and_resume_replays_it(make_llm, tmp_path, failed_task):
    tasks = ["prescribe", "reflect", "induce"]
    llm = make_llm([response()] * (3 + tasks.index(failed_task)) + [RuntimeError("connection failed")])
    method = WindowMethod(llm)
    resources = SimpleNamespace(get=lambda ids: [SimpleNamespace(cid=i, herbs=["herb-a"]) for i in ids])
    run_dir = tmp_path / "run"
    with pytest.raises(LLMRequestError):
        run_stream(method, resources, ["c1", "c2"], str(run_dir), window=1)
    assert json.loads((run_dir / "progress.json").read_text())["done"] == 1
    assert json.loads((run_dir / "memory.json").read_text())["seen"] == ["c1"]
    assert not (run_dir / "memory_final.json").exists()
    assert len((run_dir / "stream_preds.jsonl").read_text().splitlines()) == 1
    assert len((run_dir / "stream_log.jsonl").read_text().splitlines()) == 1
    failure = json.loads((run_dir / "stream_failures.jsonl").read_text())
    assert failure["done"] == 1 and failure["cids"] == ["c2"]
    assert not llm._require_responses
    n_new = 3 - tasks.index(failed_task)
    resumed = WindowMethod(make_llm([response()] * n_new, overrides=[f"llm.cache_path={llm.cfg.llm.cache_path}"]))
    run_stream(resumed, resources, ["c1", "c2"], str(run_dir), window=1)
    assert resumed.seen == ["c1", "c2"] and resumed.step == 2
    assert len(resumed.llm.requests) == n_new
    assert json.loads((run_dir / "progress.json").read_text())["done"] == 2
    assert json.loads((run_dir / "memory_final.json").read_text())["seen"] == ["c1", "c2"]


def test_failure_in_first_window_creates_no_completed_checkpoint(make_llm, tmp_path):
    llm = make_llm([RuntimeError("connection failed")])
    method = WindowMethod(llm)
    resources = SimpleNamespace(get=lambda ids: [SimpleNamespace(cid=i, herbs=["herb-a"]) for i in ids])
    with pytest.raises(LLMRequestError):
        run_stream(method, resources, ["c1"], str(tmp_path), window=1)
    assert method.seen == []
    assert not (tmp_path / "progress.json").exists()
    assert not (tmp_path / "memory_final.json").exists()


def test_legacy_stream_checkpoint_is_not_silently_resumed(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text(json.dumps({"done": 500, "total": 500, "cids_hash": "old"}))
    with pytest.raises(RuntimeError, match="--run_name"):
        check_stream_resume(str(tmp_path))
    assert json.loads(path.read_text())["done"] == 500


def test_cli_preserves_usage_when_build_aborts(make_llm, tmp_path, monkeypatch):
    llm = make_llm([response()] * 3 + [RuntimeError("connection failed")])
    resources = SimpleNamespace(
        get=lambda ids: [SimpleNamespace(cid=i, herbs=["herb-a"]) for i in ids],
        select=lambda *args: ["c1", "c2"],
    )
    monkeypatch.setattr("tcmskill.data.Resources", lambda cfg: resources)
    monkeypatch.setattr("tcmskill.llm.build_llm", lambda cfg: llm)
    monkeypatch.setattr("tcmskill.methods.build_method", lambda *args: WindowMethod(llm))
    monkeypatch.syspath_prepend(os.path.join(ROOT, "scripts"))
    monkeypatch.setattr(sys, "argv", ["run_method.py", "--method", "ours", "--run_name", "abort",
                                     "--set", f"paths.runs_dir={tmp_path}", "--set", "stream.window=1"])
    with pytest.raises(LLMRequestError):
        runpy.run_path(os.path.join(ROOT, "scripts/run_method.py"), run_name="__main__")
    run = tmp_path / "abort"
    usage = json.loads((run / "usage.json").read_text())
    assert usage["total"]["calls"] == 4 and usage["total"]["errors"] == 1
    assert json.loads((run / "progress.json").read_text())["done"] == 1
    assert not (run / "test_metrics.json").exists()


def test_cli_legacy_guard_runs_before_overwriting_metadata(tmp_path, monkeypatch):
    run = tmp_path / "legacy"
    run.mkdir()
    (run / "config.json").write_text('{"original": true}')
    (run / "progress.json").write_text('{"done": 500}')
    monkeypatch.syspath_prepend(os.path.join(ROOT, "scripts"))
    monkeypatch.setattr(sys, "argv", ["run_method.py", "--method", "ours", "--run_name", "legacy",
                                     "--set", f"paths.runs_dir={tmp_path}"])
    with pytest.raises(RuntimeError, match="--run_name"):
        runpy.run_path(os.path.join(ROOT, "scripts/run_method.py"), run_name="__main__")
    assert (run / "config.json").read_text() == '{"original": true}'
