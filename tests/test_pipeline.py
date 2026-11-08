"""End-to-end smoke test with the mock LLM (skipped if data/processed is missing)."""
import os
import sys
import json

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.chdir(ROOT)

from tcmskill.config import load_config  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data/processed/cases.jsonl")), reason="run scripts/prepare_data.py first")


@pytest.mark.parametrize("method", ["ours", "skemex_like", "case_rag", "abl_hindsight", "zeroshot", "zeroshot_cot"])
def test_methods_run_with_mock(tmp_path, method):
    from tcmskill.data import Resources
    from tcmskill.llm import build_llm
    from tcmskill.methods import build_method
    from tcmskill.runner import run_stream, run_test

    cfg = load_config("configs/default.yaml", method=method, overrides=["llm.provider=mock", "llm.cache_path=", f"paths.runs_dir={tmp_path}"])
    res = Resources(cfg)
    llm = build_llm(cfg)
    llm.call_log_path = str(tmp_path / "llm_calls.jsonl")
    m = build_method(cfg, llm, res)
    train = res.select("train", 120)
    run_stream(m, res, train, str(tmp_path), window=40)
    agg = run_test(m, res, res.select("test", 30), str(tmp_path))
    assert agg["overall"]["n"] == 30
    assert agg["overall"]["json_complete"] == 1.0
    assert os.path.exists(tmp_path / "memory_final.json")
    output = json.loads((tmp_path / "test_output_stats.json").read_text())
    assert output["counts"]["json_complete"] == 30 and output["finish_reasons"] == {"stop": 30}
    calls = [json.loads(line) for line in (tmp_path / "llm_calls.jsonl").read_text().splitlines()]
    assert calls and all(call["max_tokens"] == llm.output_budget() for call in calls)
    assert all(call["finish_reason"] == "stop" and call["attempts"] for call in calls)
    # reload memory and test again -> identical predictions (deterministic mock)
    m2 = build_method(cfg, llm, res)
    m2.load_state_dict(__import__("json").load(open(tmp_path / "memory_final.json")))
    p1 = [p.herbs for p in m.predict(res.get(res.select("test", 10)), frozen=True)]
    p2 = [p.herbs for p in m2.predict(res.get(res.select("test", 10)), frozen=True)]
    assert p1 == p2


@pytest.mark.parametrize("failed_task", ["prescribe", "reflect", "induce"])
def test_skill_build_resume_matches_uninterrupted_build(tmp_path, monkeypatch, failed_task):
    from tcmskill.data import Resources
    from tcmskill.llm import LLMRequestError, build_llm
    from tcmskill.methods import build_method
    from tcmskill.runner import run_stream

    cfg = load_config("configs/default.yaml", method="ours", overrides=["llm.provider=mock", "llm.cache_path="])
    res = Resources(cfg)
    train = res.select("train", 120)
    baseline = build_method(cfg, build_llm(cfg), res)
    run_stream(baseline, res, train, str(tmp_path / "baseline"), window=50)

    llm = build_llm(cfg)
    method = build_method(cfg, llm, res)
    original = llm.chat_json

    def fail_second_window(messages, task, **kwargs):
        if method.step == 50 and task == failed_task:
            raise LLMRequestError("simulated connection failure")
        return original(messages, task, **kwargs)

    monkeypatch.setattr(llm, "chat_json", fail_second_window)
    run = tmp_path / "resume"
    with pytest.raises(LLMRequestError):
        run_stream(method, res, train, str(run), window=50)
    assert json.loads((run / "progress.json").read_text())["done"] == 50
    assert json.loads((run / "memory.json").read_text())["step"] == 50
    resumed = build_method(cfg, build_llm(cfg), res)
    run_stream(resumed, res, train, str(run), window=50)
    assert resumed.state_dict() == baseline.state_dict()
    assert len((run / "stream_preds.jsonl").read_text().splitlines()) == 120
