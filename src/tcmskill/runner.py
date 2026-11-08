"""Stream (build / online prequential) and frozen test runners with checkpoint & resume."""
from __future__ import annotations

import logging
import os
import time
from collections import Counter

import numpy as np

from .metrics import default_subsets, evaluate_predictions, prf
from .llm import FatalLLMError, LLMRequestError
from .prescriber import Prediction
from .utils import append_jsonl, load_json, save_json, write_jsonl

LOG = logging.getLogger(__name__)
FAILURE_GUARD_VERSION = 1


def output_diagnostics(pred: Prediction) -> dict:
    """JSON syntax and prescription emptiness are separate, potentially overlapping axes."""
    attempts = pred.attempts
    return {
        "json_complete": float(pred.json_status == "complete"),
        "json_extracted": float(pred.json_status == "extracted"),
        "json_repaired_fragment": float(pred.json_status == "repaired"),
        "json_invalid": float(pred.json_status == "invalid"),
        "json_empty": float(pred.json_status == "empty"),
        "json_unknown": float(pred.json_status == "unknown" and not pred.request_failed),
        "request_failed": float(pred.request_failed),
        "empty_prescription": float(pred.empty_prescription is True),
        "truncated": float(pred.truncated),
        "had_truncation": float(pred.truncated or any(a.get("truncated") for a in attempts)),
        "had_repaired_fragment": float(pred.json_status == "repaired" or any(a.get("json_status") == "repaired" for a in attempts)),
        "json_retried": float(len(attempts) > 1),
    }


def summarize_outputs(preds: list[Prediction]) -> dict:
    counts = Counter()
    for pred in preds:
        counts.update({k: int(v) for k, v in output_diagnostics(pred).items()})
    return {
        "n": len(preds), "accepted": sum(p.ok for p in preds),
        "empty_prescription_checked": sum(p.empty_prescription is not None for p in preds),
        "counts": dict(counts), "rates": {k: v / len(preds) for k, v in counts.items()} if preds else {},
        "finish_reasons": dict(Counter(p.finish_reason or "unknown" for p in preds)),
        "first_finish_reasons": dict(Counter((p.attempts[0].get("finish_reason") if p.attempts else None) or "unknown" for p in preds)),
        "first_json_statuses": dict(Counter(p.attempts[0].get("json_status", "unknown") if p.attempts else "unknown" for p in preds)),
    }


def format_output_stats(stats: dict) -> str:
    labels = {"json_complete": "完整 JSON", "json_extracted": "从包裹文字提取完整 JSON",
              "json_repaired_fragment": "语法修复残片", "empty_prescription": "空处方",
              "json_empty": "空响应", "json_invalid": "无法解析 JSON", "request_failed": "请求失败",
              "json_unknown": "未知/非 LLM",
              "truncated": "最终输出截断", "had_truncation": "任一轮截断",
              "had_repaired_fragment": "任一轮语法修复残片", "json_retried": "重新生成过 JSON"}
    lines = ["| 输出诊断 | 例数 | 占全部病例 |", "|---|---:|---:|"]
    for key, label in labels.items():
        if key in stats["counts"] and key in stats["rates"]:
            lines.append(f"| {label} | {stats['counts'][key]} | {stats['rates'][key]:.1%} |")
        else:
            lines.append(f"| {label} | 未知 | 未知 |")
    lines.append("\nJSON 状态与空处方独立计数；空处方仅检查已解析的对象。已知请求失败单列，不混入未知/非 LLM。截断/修复统计保留重试历史，未知不等于无截断。")
    lines.append(f"最终 finish_reason：{stats['finish_reasons']}")
    return "\n".join(lines)


def save_method(method, run_dir: str, name: str = "memory.json") -> None:
    save_json(os.path.join(run_dir, name), method.state_dict(), indent=None)


def load_method(method, path: str) -> None:
    method.load_state_dict(load_json(path))


def check_stream_resume(run_dir: str) -> None:
    path = os.path.join(run_dir, "progress.json")
    if os.path.exists(path):
        if load_json(path).get("failure_guard_version") != FAILURE_GUARD_VERSION:
            raise RuntimeError(f"{run_dir} 是未启用请求失败保护的旧建库断点，不能安全续跑；请换新的 --run_name 从头建库。")
        if not os.path.exists(os.path.join(run_dir, "memory.json")):
            raise RuntimeError(f"{run_dir} 有进度但缺少 memory.json；请检查断点或换新的 --run_name。")


def _window_metrics(cases, preds) -> dict:
    vals = [prf(p.herbs, c.herbs) for c, p in zip(cases, preds) if p is not None]
    if not vals:
        return {}
    arr = np.array(vals)
    return {
        "p": float(arr[:, 0].mean()),
        "r": float(arr[:, 1].mean()),
        "f1": float(arr[:, 2].mean()),
        "len_pred": float(np.mean([len(p.herbs) for p in preds if p is not None])),
        "errors": int(sum(1 for p in preds if p is not None and not p.ok)),
        "n_ctx_skills": float(np.mean([len(p.retrieved) for p in preds if p is not None])),
        "n_used_skills": float(np.mean([len(p.used_skills) for p in preds if p is not None])),
        "output_stats": summarize_outputs([p for p in preds if p is not None]),
    }


def run_stream(method, res, cids: list[str], run_dir: str, window: int, val_ids: list[str] | None = None, val_every: int = 0, resume: bool = True, snapshot_every: int = 10) -> None:
    os.makedirs(run_dir, exist_ok=True)
    prog_path = os.path.join(run_dir, "progress.json")
    mem_path = os.path.join(run_dir, "memory.json")
    start = 0
    if resume:
        check_stream_resume(run_dir)
    if resume and os.path.exists(prog_path) and os.path.exists(mem_path):
        prog = load_json(prog_path)
        if prog.get("cids_hash") != hash_ids(cids):
            raise RuntimeError(f"{run_dir} 中已有另一份数据流的进度；请换 --run_name 或加 --fresh")
        start = prog["done"]
        load_method(method, mem_path)
        LOG.info("从断点继续：已完成 %d/%d 个病例", start, len(cids))
    elif not resume:
        for f in ("stream_log.jsonl", "stream_preds.jsonl"):
            p = os.path.join(run_dir, f)
            if os.path.exists(p):
                os.remove(p)
    for s in range(start, len(cids), window):
        t0 = time.time()
        batch = res.get(cids[s : s + window])
        try:
            # Only commit a window after all of its learning requests have completed.
            with method.llm.require_responses():
                if method.needs_attempt:
                    preds = method.predict(batch, frozen=False)
                    if any(p.request_failed for p in preds):
                        raise LLMRequestError("A build prediction failed before observation")
                else:
                    preds = [None] * len(batch)
                obs = method.observe(batch, preds)
                log = {"window": method.window - 1, "done": s + len(batch), **_window_metrics(batch, preds), "memory": method.summary(), "observe": obs}
                if val_ids and val_every and method.window % val_every == 0:
                    vp = method.predict(res.get(val_ids), frozen=True)
                    log["val_f1"] = float(np.mean([prf(p.herbs, res.cases[p.cid].herbs)[2] for p in vp]))
        except (LLMRequestError, FatalLLMError) as e:
            append_jsonl(os.path.join(run_dir, "stream_failures.jsonl"), [{
                "t": time.time(), "done": s, "cids": [c.cid for c in batch],
                "error": str(e), "usage": method.llm.usage.summary(),
            }])
            LOG.error("建库请求失败，本窗口未提交，进度保留在 %d/%d。网络恢复后重新运行同一命令续跑；不要在当前进程继续使用部分更新的记忆。", s, len(cids))
            raise
        log["usage"] = method.llm.usage.summary()["total"]
        log["secs"] = round(time.time() - t0, 1)
        append_jsonl(os.path.join(run_dir, "stream_log.jsonl"), [log])
        append_jsonl(os.path.join(run_dir, "stream_preds.jsonl"), [p.to_dict() for p in preds if p is not None])
        save_method(method, run_dir)
        if snapshot_every and method.window % snapshot_every == 0:
            save_method(method, run_dir, f"memory_w{method.window:04d}.json")
        save_json(prog_path, {"done": s + len(batch), "total": len(cids), "cids_hash": hash_ids(cids),
                              "failure_guard_version": FAILURE_GUARD_VERSION})
        LOG.info(
            "window %d | 已学 %d/%d | 本窗 F1 %s | 记忆 %s | 累计费用 $%.3f | %.0fs",
            method.window - 1,
            s + len(batch),
            len(cids),
            f"{log['f1']:.3f}" if "f1" in log else "—",
            log["memory"],
            log["usage"]["cost_usd"],
            log["secs"],
        )
    save_method(method, run_dir, "memory_final.json")


def run_test(method, res, cids: list[str], run_dir: str, name: str = "test") -> dict:
    os.makedirs(run_dir, exist_ok=True)
    cases = res.get(cids)
    preds = method.predict(cases, frozen=True)
    write_jsonl(os.path.join(run_dir, f"{name}_preds.jsonl"), [p.to_dict() for p in preds])
    return evaluate_run(res, preds, run_dir, name)


def evaluate_run(res, preds: list[Prediction], run_dir: str, name: str = "test") -> dict:
    pred_map = {p.cid: p.herbs for p in preds}
    extra = {p.cid: {"oov": float(len(p.oov) > 0), "error": float(not p.ok), "n_used_skills": float(len(p.used_skills)), **output_diagnostics(p)} for p in preds}
    out = evaluate_predictions(pred_map, res.cases, res.herb_info, res.incompat, default_subsets(res.cases, pred_map), extra=extra)
    save_json(os.path.join(run_dir, f"{name}_metrics.json"), out["aggregate"])
    save_json(os.path.join(run_dir, f"{name}_output_stats.json"), summarize_outputs(preds))
    write_jsonl(os.path.join(run_dir, f"{name}_per_case.jsonl"), ({"cid": k, **v} for k, v in out["per_case"].items()))
    return out["aggregate"]


def hash_ids(ids: list[str]) -> str:
    import hashlib

    return hashlib.md5("|".join(ids).encode()).hexdigest()


def format_metrics(agg: dict, keys=("f1", "p", "r", "soft_f1", "p@5", "r@10", "ndcg@10", "len_pred", "len_gt", "len_abs_err", "unsafe", "gt_unsafe", "oov", "error")) -> str:
    lines = []
    head = "| 子集 | n | " + " | ".join(keys) + " |"
    lines.append(head)
    lines.append("|" + "---|" * (len(keys) + 2))
    for sub, m in agg.items():
        if not m.get("n"):
            continue
        cells = []
        for k in keys:
            v = m.get(k)
            cells.append(f"{v:.3f}" if isinstance(v, float) else ("—" if v is None else str(v)))
        lines.append(f"| {sub} | {m['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
