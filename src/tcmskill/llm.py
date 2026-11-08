"""LLM clients.

* DeepSeekLLM – OpenAI-compatible chat completions (DeepSeek by default). Uses the `openai`
  SDK if installed, else plain urllib. Features: sqlite response cache (identical request =
  paid once, also makes runs resumable/reproducible), retries with backoff, automatic
  degradation if the endpoint rejects `thinking`/`response_format`, robust JSON parsing,
  token/cost accounting (cache-hit / cache-miss / output tokens).
* MockLLM (see mock.py) – deterministic fake backend for smoke tests and cost estimation.

All callers use `llm.chat_json(messages, task=..., meta=...)`. Real backends use `meta`
only to correlate logs with cases; it is never sent to the provider.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

LOG = logging.getLogger(__name__)
OUTPUT_PROTOCOL_VERSION = 2


@dataclass
class LLMResult:
    text: str = ""
    data: Any = None
    ok: bool = False
    error: str | None = None
    cached: bool = False
    model: str = ""
    usage: dict = field(default_factory=dict)
    latency: float = 0.0
    reasoning: str = ""
    finish_reason: str | None = None
    json_status: str = "unknown"
    truncated: bool = False
    request_key: str = ""
    max_tokens: int = 0
    attempts: list[dict] = field(default_factory=list)
    request_failed: bool = False
    error_chain: list[dict] = field(default_factory=list)


def _error_chain(error: Exception) -> list[dict]:
    chain, seen = [], set()
    while error is not None and id(error) not in seen and len(chain) < 8:
        seen.add(id(error))
        chain.append({"type": type(error).__name__, "message": str(error)[:500]})
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return chain


# ============================================================================ json parsing
def parse_json_with_status(text: str) -> tuple[Any, str]:
    """Distinguish intact JSON, extracted JSON, and syntactically repaired fragments."""
    t = (text or "").strip()
    if not t:
        return None, "empty"
    try:
        obj = json.loads(t)
        return (obj, "complete") if isinstance(obj, (dict, list)) else (None, "invalid")
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, (dict, list)):
                return obj, "extracted"
        except Exception:
            pass
    i, j = t.find("{"), t.rfind("}")
    if not m and not t.startswith(("{", "[")) and i != -1 and j > i:
        try:
            obj = json.loads(t[i : j + 1])
            if isinstance(obj, (dict, list)):
                return obj, "extracted"
        except Exception:
            pass
    try:  # optional dependency
        import json_repair  # type: ignore

        obj = json_repair.loads(m.group(1) if m else t)
        return (obj, "repaired") if isinstance(obj, (dict, list)) else (None, "invalid")
    except Exception:
        return None, "invalid"


def parse_json(text: str) -> Any:
    return parse_json_with_status(text)[0]


# ============================================================================ usage
class UsageTracker:
    def __init__(self, prices: dict, model: str):
        self.prices = prices or {}
        self.model = model
        self.lock = threading.Lock()
        self.by_task: dict[str, dict] = {}
        self.output_by_task: dict[str, dict] = {}
        self.models_seen: dict[str, int] = {}

    def _price(self, model: str) -> dict:
        for k in (model, self.model):
            if k in self.prices:
                return self.prices[k]
        return self.prices.get("default", {"hit": 0.0, "miss": 0.0, "out": 0.0})

    def add(self, task: str, usage: dict | None, cached: bool, error: bool, model: str = "") -> None:
        usage = usage or {}
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = int(usage.get("prompt_cache_miss_tokens", usage.get("prompt_tokens", 0) - hit) or 0)
        out = int(usage.get("completion_tokens", 0) or 0)
        reas = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
        pr = self._price(model or self.model)
        cost = 0.0 if cached else (hit * pr.get("hit", 0) + miss * pr.get("miss", 0) + out * pr.get("out", 0)) / 1e6
        with self.lock:
            s = self.by_task.setdefault(task, {"calls": 0, "cached": 0, "errors": 0, "hit": 0, "miss": 0, "out": 0, "reasoning": 0, "cost_usd": 0.0})
            s["calls"] += 1
            s["cached"] += int(cached)
            s["errors"] += int(error)
            if not cached:
                s["hit"] += hit
                s["miss"] += miss
                s["out"] += out
                s["reasoning"] += reas
                s["cost_usd"] += cost
            if model:
                self.models_seen[model] = self.models_seen.get(model, 0) + 1

    def record_output(self, task: str, res: LLMResult) -> None:
        first = res.attempts[0] if res.attempts else {}
        with self.lock:
            out = self.output_by_task.setdefault(task, {
                "requests": 0, "cached": 0, "accepted_json": 0,
                "json_statuses": {}, "finish_reasons": {},
                "first_json_statuses": {}, "first_finish_reasons": {},
                "had_truncation": 0, "had_repaired_fragment": 0,
                "request_failures": 0,
            })
            out["requests"] += 1
            out["cached"] += int(res.cached)
            out["accepted_json"] += int(res.ok)
            out["request_failures"] += int(res.request_failed)
            out["had_truncation"] += int(res.truncated or any(a.get("truncated") for a in res.attempts))
            out["had_repaired_fragment"] += int(res.json_status == "repaired" or any(a.get("json_status") == "repaired" for a in res.attempts))
            for field, value in (("json_statuses", res.json_status), ("finish_reasons", res.finish_reason),
                                 ("first_json_statuses", first.get("json_status")),
                                 ("first_finish_reasons", first.get("finish_reason"))):
                value = value or "unknown"
                out[field][value] = out[field].get(value, 0) + 1

    def summary(self) -> dict:
        with self.lock:
            tot = {"calls": 0, "cached": 0, "errors": 0, "hit": 0, "miss": 0, "out": 0, "reasoning": 0, "cost_usd": 0.0}
            for s in self.by_task.values():
                for k in tot:
                    tot[k] += s[k]
            return {"total": tot, "by_task": json.loads(json.dumps(self.by_task)),
                    "output_by_task": json.loads(json.dumps(self.output_by_task)), "models_seen": dict(self.models_seen)}

    def pretty(self) -> str:
        s = self.summary()
        t = s["total"]
        lines = [
            f"LLM 调用 {t['calls']} 次（缓存命中 {t['cached']}，失败 {t['errors']}）；"
            f"输入 token：命中 {t['hit']:,} / 未命中 {t['miss']:,}；输出 {t['out']:,}（其中思考 {t['reasoning']:,}）；"
            f"费用约 ${t['cost_usd']:.3f}"
        ]
        for k, v in sorted(s["by_task"].items()):
            lines.append(f"  - {k}: {v['calls']} 次, ${v['cost_usd']:.3f}")
        if s["models_seen"]:
            lines.append(f"  实际模型版本: {s['models_seen']}")
        return "\n".join(lines)


# ============================================================================ cache
class SQLiteCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, ts REAL)")
        self.conn.commit()

    def get(self, key: str) -> dict | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict) -> None:
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, json.dumps(value, ensure_ascii=False), time.time()))
            self.conn.commit()


# ============================================================================ base
class BaseLLM:
    name = "base"

    def __init__(self, cfg):
        self.cfg = cfg
        lc = cfg.llm
        self.model = lc.model
        self.workers = int(lc.workers)
        self.max_tokens = int(lc.max_tokens)
        self.thinking = bool(lc.thinking)
        self.thinking_max_tokens = int(lc.get("thinking_max_tokens", 16000))
        self.usage = UsageTracker(dict(lc.get("prices", {})), lc.model)
        self.call_log_path: str | None = None
        self._log_lock = threading.Lock()
        self._require_responses = False

    @contextmanager
    def require_responses(self):
        """Do not let an exhausted API request silently advance a learning window."""
        previous = self._require_responses
        self._require_responses = True
        try:
            yield
        finally:
            self._require_responses = previous

    def chat_json(self, messages: list[dict], task: str, meta: dict | None = None, max_tokens: int | None = None) -> LLMResult:
        raise NotImplementedError

    def output_budget(self, max_tokens: int | None = None) -> int:
        budget = self.max_tokens if max_tokens is None else int(max_tokens)
        if budget <= 0:
            raise ValueError("max_tokens must be positive")
        return max(budget, self.thinking_max_tokens) if self.thinking else budget

    def _log_call(self, task: str, res: LLMResult, meta: dict | None = None) -> None:
        self.usage.record_output(task, res)
        if not self.call_log_path or not self.cfg.llm.get("log_calls", True):
            return
        meta = meta or {}
        rec = {"t": round(time.time(), 2), "task": task, "ok": res.ok, "cached": res.cached, "model": res.model, "latency": round(res.latency, 2), "usage": res.usage, "error": res.error,
               "protocol_version": OUTPUT_PROTOCOL_VERSION, "request_key": res.request_key,
               "cid": meta.get("cid"), "cids": meta.get("cids", []),
               "max_tokens": res.max_tokens, "finish_reason": res.finish_reason,
               "json_status": res.json_status, "truncated": res.truncated, "attempts": res.attempts,
               "request_failed": res.request_failed, "error_chain": res.error_chain}
        with self._log_lock:
            with open(self.call_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def map(self, fn: Callable[[Any], Any], items: Sequence[Any], desc: str = "", workers: int | None = None) -> list:
        """Run fn over items concurrently; preserves order; logs progress."""
        items = list(items)
        if not items:
            return []
        workers = max(1, min(workers or self.workers, len(items)))
        out: list = [None] * len(items)
        done = 0
        t0 = time.time()
        step = max(1, len(items) // 10)
        if workers == 1:
            for i, it in enumerate(items):
                out[i] = fn(it)
            return out
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
            try:
                for fut in as_completed(futs):
                    i = futs[fut]
                    out[i] = fut.result()
                    done += 1
                    if desc and (done % step == 0 or done == len(items)) and len(items) >= 200:
                        LOG.info("%s: %d/%d (%.0fs)", desc, done, len(items), time.time() - t0)
            except BaseException:
                for fut in futs:
                    fut.cancel()
                raise
        return out


# ============================================================================ DeepSeek / OpenAI-compatible
class FatalLLMError(RuntimeError):
    """Errors that make every further call pointless (bad key, no balance, unknown model)."""


class LLMRequestError(RuntimeError):
    """An API request failed during a response-required learning phase."""


class LLMHTTPError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.body = body


class DeepSeekLLM(BaseLLM):
    name = "deepseek"

    def __init__(self, cfg, send_thinking: bool = True):
        super().__init__(cfg)
        lc = cfg.llm
        self.api_key = os.environ.get(lc.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(
                f"环境变量 {lc.api_key_env} 未设置。请先 export {lc.api_key_env}=sk-...（或用 --llm mock 跑冒烟测试）"
            )
        self.base_url = lc.base_url.rstrip("/")
        self.timeout = float(lc.timeout)
        self.max_retries = int(lc.max_retries)
        if self.max_retries < 0:
            raise ValueError("max_retries must be nonnegative")
        self._transport = threading.local()
        self.temperature = float(lc.temperature)
        self.effort = lc.get("reasoning_effort")
        self.json_mode = bool(lc.json_mode)
        self.json_repair_retries = int(lc.get("json_repair_retries", 1))
        if self.json_repair_retries < 0:
            raise ValueError("json_repair_retries must be nonnegative")
        self.accept_repaired_json = bool(lc.get("accept_repaired_json", False))
        self.send_thinking = send_thinking
        self._dropped: set[str] = set()
        self.cache = SQLiteCache(lc.cache_path) if lc.get("cache_path") else None
        self._sdk = None
        try:
            import openai  # type: ignore

            self._sdk = openai.OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout, max_retries=0)
        except Exception:
            LOG.info("openai SDK 不可用，使用 urllib 直接请求 %s", self.base_url)

    # ---------------------------------------------------------------- request
    def _payload(self, messages: list[dict], max_tokens: int) -> tuple[dict, dict]:
        max_tokens = self.output_budget(max_tokens)
        std: dict = {"model": self.model, "messages": messages, "max_tokens": max_tokens, "stream": False}
        extra: dict = {}
        if self.json_mode and "response_format" not in self._dropped:
            std["response_format"] = {"type": "json_object"}
        if self.send_thinking and "thinking" not in self._dropped:
            extra["thinking"] = {"type": "enabled" if self.thinking else "disabled"}
        if self.thinking:
            if self.effort and "reasoning_effort" not in self._dropped:
                extra["reasoning_effort"] = self.effort
        else:
            std["temperature"] = self.temperature
        return std, extra

    def _post(self, std: dict, extra: dict) -> dict:
        if self._sdk is not None:
            import openai  # type: ignore

            try:
                resp = self._sdk.chat.completions.with_raw_response.create(**std, extra_body=extra or None)
            except openai.APIStatusError as e:  # type: ignore[attr-defined]
                body = ""
                try:
                    body = e.response.text
                except Exception:
                    body = str(e)
                raise LLMHTTPError(e.status_code, body) from e
            # Compatible providers can use different types for optional metadata.
            # Read JSON directly instead of round-tripping through SDK models.
            return resp.http_response.json()
        payload = dict(std)
        payload.update(extra)
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise LLMHTTPError(e.code, e.read().decode("utf-8", "replace")) from e

    def _maybe_degrade(self, err: LLMHTTPError) -> bool:
        if err.status != 400:
            return False
        body = err.body.lower()
        for p in ("thinking", "reasoning_effort", "response_format"):
            if p in body and p not in self._dropped:
                LOG.warning("端点不接受参数 %s，之后的请求将不再发送它（%s）", p, err.body[:200])
                self._dropped.add(p)
                return True
        return False

    def _call(self, messages: list[dict], max_tokens: int) -> dict:
        last: Exception | None = None
        history = []
        self._transport.attempts = history
        for attempt in range(self.max_retries + 1):
            std, extra = self._payload(messages, max_tokens)
            started = time.time()
            try:
                raw = self._post(std, extra)
            except Exception as e:
                last = e
            else:
                history.append({"attempt": attempt + 1, "ok": True, "latency": round(time.time() - started, 2)})
                return raw
            entry = {"attempt": attempt + 1, "ok": False, "latency": round(time.time() - started, 2),
                     "error_chain": _error_chain(last)}
            history.append(entry)
            if isinstance(last, LLMHTTPError):
                e = last
                entry["http_status"] = e.status
                if self._maybe_degrade(e):
                    entry["dropped_parameters"] = sorted(self._dropped)
                    continue
                if e.status in (401, 402, 403):
                    hint = {401: "API key 无效", 402: "账户余额不足", 403: "无权限"}[e.status]
                    raise FatalLLMError(f"{hint}（HTTP {e.status}）：{e.body[:200]}") from e
                if e.status == 404 or (e.status == 400 and "model" in e.body.lower() and "not" in e.body.lower()):
                    raise FatalLLMError(f"模型 {self.model} 不可用（HTTP {e.status}）：{e.body[:200]}。请检查 llm.model（2026-11 可用：deepseek-flash / deepseek-v4-pro）") from e
                if e.status in (400, 422):
                    raise e
            detail = " <- ".join(f"{e['type']}: {e['message']}" for e in entry["error_chain"])
            if attempt == self.max_retries:
                LOG.error("LLM 请求失败（第 %d 次），重试已耗尽：%s", attempt + 1, detail)
                break
            wait = min(60.0, 2.0**attempt) + random.random()
            entry["retry_delay"] = round(wait, 2)
            LOG.warning("LLM 请求失败（第 %d 次）：%s；%.1fs 后重试", attempt + 1, detail, wait)
            time.sleep(wait)
        raise RuntimeError(f"LLM 请求多次失败：{last}") from last

    # ---------------------------------------------------------------- public
    def _key(self, messages: list[dict], max_tokens: int) -> str:
        blob = json.dumps(
            {"version": OUTPUT_PROTOCOL_VERSION, "endpoint": self.base_url,
             "m": self.model, "msg": messages, "mt": self.output_budget(max_tokens),
             "t": self.temperature, "th": self.thinking, "send_thinking": self.send_thinking,
             "ef": self.effort if self.thinking else None, "js": self.json_mode,
             "json_repair_retries": self.json_repair_retries, "accept_repaired_json": self.accept_repaired_json},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat_json(self, messages: list[dict], task: str, meta: dict | None = None, max_tokens: int | None = None) -> LLMResult:
        max_tokens = self.output_budget(max_tokens)
        key = self._key(messages, max_tokens)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit and hit.get("version") == OUTPUT_PROTOCOL_VERSION and hit.get("result", {}).get("ok"):
                res = LLMResult(**hit["result"])
                res.cached, res.latency = True, 0.0
                self.usage.add(task, res.usage, cached=True, error=not res.ok, model=res.model)
                self._log_call(task, res, meta)
                return res
        t0 = time.time()
        attempts = []
        request_messages = messages
        fatal = None
        request_error = None
        for attempt in range(self.json_repair_retries + 1):
            res = LLMResult(request_key=key, max_tokens=max_tokens)
            started = time.time()
            request_failed = False
            self._transport.attempts = []
            try:
                raw = self._call(request_messages, max_tokens)
                choice = raw["choices"][0]
                msg = choice["message"]
                res.text = msg.get("content") or ""
                res.reasoning = msg.get("reasoning_content") or ""
                res.model = raw.get("model", self.model)
                res.usage = raw.get("usage") or {}
                res.finish_reason = choice.get("finish_reason")
                res.truncated = res.finish_reason == "length"
                res.data, res.json_status = parse_json_with_status(res.text)
                if res.truncated:
                    res.error = "truncated_output"
                elif res.data is None:
                    res.error = "invalid_json"
                elif res.json_status == "repaired" and not self.accept_repaired_json:
                    res.error = "repaired_json_rejected"
                else:
                    res.ok = True
            except Exception as e:
                request_failed = True
                request_error = e
                res.request_failed = True
                res.error_chain = _error_chain(e)
                res.error = f"{type(e).__name__}: {e}"[:500]
                if isinstance(e, FatalLLMError):
                    fatal = e
            self.usage.add(task, res.usage, cached=False, error=not res.ok, model=res.model)
            attempts.append({"attempt": attempt + 1, "ok": res.ok, "model": res.model,
                             "finish_reason": res.finish_reason, "json_status": res.json_status,
                             "truncated": res.truncated, "max_tokens": max_tokens,
                             "usage": res.usage, "error": res.error, "text_chars": len(res.text),
                             "request_failed": res.request_failed, "error_chain": res.error_chain,
                             "transport_attempts": self._transport.attempts,
                             "latency": round(time.time() - started, 2),
                             "dropped_parameters": sorted(self._dropped)})
            if res.ok or request_failed:
                break
            # A repaired fragment is not a completed answer. Retry under the same budget.
            request_messages = [dict(message) for message in messages]
            retry_instruction = "上一轮回答被截断或不是完整 JSON。请独立重新回答本任务，输出一个完整的 JSON 对象；精简文字，不要省略必需字段，不要输出其他文字。"
            if request_messages and request_messages[-1]["role"] == "user" and isinstance(request_messages[-1].get("content"), str):
                request_messages[-1]["content"] += "\n\n" + retry_instruction
            else:
                request_messages.append({"role": "user", "content": retry_instruction})
        res.attempts = attempts
        res.latency = time.time() - t0
        if res.ok and self.cache is not None:
            cached_result = asdict(res)
            cached_result["reasoning"] = res.reasoning[:2000]
            self.cache.set(key, {"version": OUTPUT_PROTOCOL_VERSION, "result": cached_result})
        self._log_call(task, res, meta)
        if fatal is not None:
            raise fatal
        if res.request_failed and self._require_responses:
            raise LLMRequestError(f"{task} request {key[:12]} failed: {res.error}") from request_error
        return res


def build_llm(cfg) -> BaseLLM:
    p = cfg.llm.provider
    if p == "mock":
        from .mock import MockLLM

        return MockLLM(cfg)
    if p == "deepseek":
        return DeepSeekLLM(cfg, send_thinking=True)
    if p == "openai_compatible":
        return DeepSeekLLM(cfg, send_thinking=False)
    raise ValueError(f"unknown llm.provider {p}")
