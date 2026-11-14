"""CLI-only progress observation; no changes to prompts, checkpoints or experiment identity."""
from __future__ import annotations

import contextlib
import math
import shutil
import sys
import threading
import time
import unicodedata

from .knowledge_evolution import KnowledgeEvolver


TASK_NAMES = {
    "knowledge_rx": "开方", "knowledge_opt": "总结技能", "knowledge_applicability": "匹配技能",
    "knowledge_skill_review": "审核技能", "knowledge_branch_conditions": "复核分支",
    "knowledge_scope_review": "复核范围", "knowledge_skill_support": "核查支持",
}
MAP_NAMES = {
    "knowledge_solutions": "生成方案", "knowledge_rx": "生成方案", "practice_control": "配对基线",
    "practice_treatment": "配对试用", "fresh_gate_old": "门控基线", "fresh_gate_new": "门控技能",
    "fresh_confirm_old": "复核基线", "fresh_confirm_new": "复核技能",
}


def fit_line(text, width):
    result, used = [], 0
    for char in text:
        size = 0 if unicodedata.combining(char) else 2 if unicodedata.east_asian_width(char) in ("W", "F") else 1
        if used + size > width:
            break
        result.append(char)
        used += size
    return "".join(result)


class TrainingProgress:
    def __init__(self, enabled=True, stream=None, interval=None, label="训练"):
        self.label = label
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = enabled
        try:
            self.tty = bool(getattr(self.stream, "isatty", lambda: False)())
        except (OSError, ValueError):
            self.tty, self.enabled = False, False
        self.interval = interval if interval is not None else (0.5 if self.tty else 10.0)
        self.lock, self.stop = threading.RLock(), threading.Event()
        self.thread = None
        self.started, self.last_render = time.monotonic(), float("-inf")
        self.total, self.done, self.batch, self.batches = None, 0, 0, 0
        self.phase, self.mapping = f"准备{label}", None
        self.active, self.next_request, self.finished, self.cached = {}, 0, 0, 0

    def __enter__(self):
        if self.enabled:
            self.refresh(force=True)
            self.thread = threading.Thread(target=self._heartbeat, name="knowledge-progress", daemon=True)
            self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        if self.thread:
            self.thread.join()
        if exc_type is not None:
            self.phase = f"{self.label}中断，未完成的批次不计入进度"
        self.refresh(force=True, final=True)

    def _heartbeat(self):
        while not self.stop.wait(0.5):
            self.refresh()

    def set_phase(self, phase):
        with self.lock:
            self.phase = phase
            self.refresh(force=True)

    def configure(self, total, done, step, batch_size):
        with self.lock:
            self.total, self.done = total, done
            self.batches = math.ceil(total / batch_size)
            self.batch = step
            self.refresh(force=True)

    def begin_step(self, done, step):
        with self.lock:
            self.done, self.batch = done, step
            self.mapping = None
            self.phase = "检索知识与匹配技能"
            self.refresh(force=True)

    def finish(self, done, step, complete):
        with self.lock:
            self.done, self.batch = done, step
            self.mapping = None
            self.phase = f"{self.label}完成" if complete else f"本轮结束，尚有未{self.label}病例"
            self.refresh(force=True)

    def _line(self):
        now = time.monotonic()
        if self.total is None:
            heading = self.label
        else:
            filled = min(10, int(10 * self.done / max(1, self.total)))
            heading = f"{self.label} [{'#' * filled}{'-' * (10 - filled)}] {self.done}/{self.total}例 批次{self.batch}/{self.batches}"
        if self.mapping:
            detail = f"{self.mapping['name']} {self.mapping['done']}/{self.mapping['total']}"
        else:
            detail = self.phase
        if self.active:
            request = min(self.active.values(), key=lambda r: r["started"])
            detail += f" | {request['task']}等待{int(now - request['started'])}s"
            if request["cid"]:
                detail += f" {request['cid']}"
            if len(self.active) > 1:
                detail += f" ({len(self.active)}请求并行)"
        else:
            detail += f" | 已完成请求{self.finished} 缓存{self.cached} 用时{int(now - self.started)}s"
        return f"{heading} | {detail}"

    def refresh(self, force=False, final=False):
        with self.lock:
            if not self.enabled or (not force and time.monotonic() - self.last_render < self.interval):
                return
            line = self._line()
            if self.tty:
                line = "\r\033[2K" + fit_line(line, max(20, shutil.get_terminal_size((120, 20)).columns - 1))
            try:
                self.stream.write(line + ("\n" if final or not self.tty else ""))
                self.stream.flush()
            except (OSError, ValueError):
                # A closed display must not turn a successful model call into a training failure.
                self.enabled = False
            self.last_render = time.monotonic()

    @contextlib.contextmanager
    def request(self, task, meta):
        with self.lock:
            self.next_request += 1
            key = self.next_request
            self.active[key] = {"task": TASK_NAMES.get(task, task), "cid": str((meta or {}).get("cid", ""))[:24],
                                "started": time.monotonic()}
            self.refresh()
        result = []
        try:
            yield result
        finally:
            with self.lock:
                self.active.pop(key, None)
                if result:
                    self.finished += 1
                    self.cached += int(bool(getattr(result[0], "cached", False)))
                self.refresh()

    @contextlib.contextmanager
    def mapped(self, desc, total):
        with self.lock:
            previous = self.mapping
            record = {"name": MAP_NAMES.get(desc, desc or "处理方案"), "done": 0, "total": total}
            self.mapping = record
            self.refresh(force=True)
        try:
            yield record
        finally:
            with self.lock:
                self.refresh(force=True)
                self.mapping = previous

    def item_finished(self, record):
        with self.lock:
            record["done"] += 1
            self.refresh()


class ProgressLLM:
    """Delegate exactly the same requests and ordered maps, observing their lifetime only."""
    def __init__(self, llm, progress):
        self.llm, self.progress = llm, progress

    def __getattr__(self, name):
        return getattr(self.llm, name)

    def chat_json(self, messages, task, meta=None, max_tokens=None):
        with self.progress.request(task, meta) as result:
            answer = self.llm.chat_json(messages, task, meta=meta, max_tokens=max_tokens)
            result.append(answer)
            return answer

    def map(self, fn, items, desc="", workers=None):
        items = list(items)
        with self.progress.mapped(desc, len(items)) as record:
            def observe(item):
                answer = fn(item)
                self.progress.item_finished(record)
                return answer
            return self.llm.map(observe, items, desc=desc, workers=workers)


class ProgressKnowledgeEvolver(KnowledgeEvolver):
    def __init__(self, *args, progress, **kwargs):
        self.progress = progress
        super().__init__(*args, **kwargs)

    def train(self, max_steps=None):
        self.progress.configure(len(self.partition["practice"]), self.cursor, self.step, int(self.kc.batch))
        result = super().train(max_steps)
        self.progress.finish(self.cursor, self.step, result["complete"])
        return result

    def run_step(self):
        # The previous step has committed before the next one starts; never count a failed batch.
        self.progress.begin_step(self.cursor, self.step + 1)
        return super().run_step()

    def optimize(self, payload):
        self.progress.set_phase("对比方案与修订技能")
        return super().optimize(payload)

    def _advance_libraries(self, groups, row):
        self.progress.set_phase("审核技能与独立验证")
        return super()._advance_libraries(groups, row)
