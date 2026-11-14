"""Progress must stay visible without changing requests, ordering or resume semantics."""
import io
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmskill.config import load_config  # noqa: E402
from tcmskill.knowledge_evolution import KnowledgeEvolver  # noqa: E402
from tcmskill.knowledge_progress import ProgressKnowledgeEvolver, ProgressLLM, TrainingProgress, fit_line  # noqa: E402
from tcmskill.llm import build_llm  # noqa: E402


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("workers", [1, 4])
def test_serial_and_parallel_requests_and_map_order_are_unchanged(monkeypatch, workers):
    cfg = load_config(str(ROOT / "configs/default.yaml"), overrides=["llm.provider=mock", "llm.cache_path="])
    base = build_llm(cfg)
    calls, stream = [], io.StringIO()

    def answer(messages, task, meta=None, max_tokens=None):
        calls.append((messages, task, meta, max_tokens))
        time.sleep(0.002 * (5 - meta["index"]))
        return SimpleNamespace(cached=meta["index"] % 2 == 0, index=meta["index"])

    monkeypatch.setattr(base, "chat_json", answer)
    with TrainingProgress(stream=stream, interval=0) as progress:
        progress.configure(8, 0, 0, 8)
        progress.begin_step(0, 1)
        client = ProgressLLM(base, progress)
        assert client.model == base.model and client.cfg is base.cfg
        messages = [{"role": "user", "content": "unchanged prompt"}]
        results = client.map(lambda i: client.chat_json(messages, "knowledge_rx", meta={"index": i}, max_tokens=91),
                             range(4), desc="knowledge_solutions", workers=workers)
        assert [r.index for r in results] == list(range(4))
        assert sorted(c[2]["index"] for c in calls) == list(range(4))
        assert all(c[0] is messages and c[1] == "knowledge_rx" and c[3] == 91 for c in calls)
        assert progress.finished == 4 and progress.cached == 2
        assert not progress.active and progress.mapping is None and progress.done == 0
        progress.finish(8, 1, True)
    output = stream.getvalue()
    assert "生成方案 4/4" in output and "8/8例" in output and "训练完成" in output
    assert "\033" not in output and "\r" not in output
    assert not progress.thread.is_alive()


def test_heartbeat_displays_a_still_running_request_without_fake_completion():
    stream = Terminal()
    with TrainingProgress(stream=stream, interval=0) as progress:
        progress.configure(64, 8, 1, 8)
        progress.begin_step(8, 2)
        with progress.request("knowledge_opt", {"cid": "H1"}):
            initial = len(stream.getvalue())
            time.sleep(0.65)
            assert len(stream.getvalue()) > initial
            assert "总结技能等待" in stream.getvalue()
            assert progress.done == 8 and progress.finished == 0
        progress.finish(8, 1, False)
    assert "8/64例" in stream.getvalue() and "尚有未训练病例" in stream.getvalue()
    assert "\r\033[2K" in stream.getvalue() and stream.getvalue().endswith("\n")


def test_failed_request_does_not_advance_batch_and_cleans_display():
    class BrokenLLM:
        def chat_json(self, *args, **kwargs):
            raise RuntimeError("API failed")

    stream = io.StringIO()
    progress = TrainingProgress(stream=stream, interval=0)
    with pytest.raises(RuntimeError, match="API failed"), progress:
        progress.configure(16, 8, 1, 8)
        progress.begin_step(8, 2)
        ProgressLLM(BrokenLLM(), progress).chat_json([], "knowledge_rx")
    assert not progress.active and progress.finished == 0 and progress.done == 8
    assert "训练中断" in stream.getvalue() and "16/16例" not in stream.getvalue()
    assert not progress.thread.is_alive()


def test_disabled_progress_is_silent_and_does_not_start_thread():
    stream = io.StringIO()
    with TrainingProgress(enabled=False, stream=stream) as progress:
        progress.configure(2, 0, 0, 2)
        progress.set_phase("审核")
        with progress.request("knowledge_skill_review", None) as result:
            result.append(SimpleNamespace(cached=False))
        progress.finish(2, 1, True)
    assert stream.getvalue() == "" and progress.thread is None and progress.finished == 1


def test_closed_progress_stream_cannot_fail_training():
    stream = io.StringIO()
    stream.close()
    with TrainingProgress(stream=stream) as progress:
        progress.configure(2, 0, 0, 2)
        progress.finish(2, 1, True)
    assert not progress.enabled


def test_terminal_width_accounts_for_wide_characters():
    assert fit_line("训练 abc", 4) == "训练"
    assert fit_line("训练 abc", 6) == "训练 a"
    assert fit_line("e\u0301x", 1) == "e\u0301"


def test_stdout_remains_machine_readable(capsys):
    with TrainingProgress(interval=0) as progress:
        progress.configure(8, 8, 1, 8)
        progress.finish(8, 1, True)
    print(json.dumps({"complete": True}))
    output = capsys.readouterr()
    assert json.loads(output.out) == {"complete": True} and "8/8例" in output.err


def test_resume_starts_at_committed_cursor_and_max_steps_is_not_full_completion(monkeypatch):
    stream = io.StringIO()
    progress = TrainingProgress(stream=stream, interval=0)
    exp = ProgressKnowledgeEvolver.__new__(ProgressKnowledgeEvolver)
    exp.progress, exp.partition, exp.kc = progress, {"practice": list(range(64))}, SimpleNamespace(batch=8)
    exp.cursor, exp.step = 8, 1

    def step(self):
        assert progress.done == 8 and progress.batch == 2
        self.cursor, self.step = 16, 2
        assert progress.done == 8  # Not advanced by an uncommitted run_step result.
        return {"step": 2}

    def train(self, max_steps):
        assert max_steps == 2
        self.run_step()
        return {"complete": False, "seen_cases": self.cursor}

    monkeypatch.setattr(KnowledgeEvolver, "run_step", step)
    monkeypatch.setattr(KnowledgeEvolver, "train", train)
    with progress:
        summary = exp.train(2)
    assert summary == {"complete": False, "seen_cases": 16}
    assert progress.done == 16 and "8/64例" in stream.getvalue() and "16/64例" in stream.getvalue()
    assert "64/64例" not in stream.getvalue()
