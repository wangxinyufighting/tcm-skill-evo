"""决策层（零 API 费用）：在已有 LLM 处方上做"唯一先例照抄 + kNN 融合 / 味数校准"，参数只在验证集上调。

  # 默认：5 折交叉拟合 —— 每个病例都用"其余 4 折"调出的参数处理，报告全部病例（无信息泄漏）
  python scripts/posthoc.py --run runs/s2_caserag_k5_nothink_val1000 --split val

  # 指定调参段 / 报告段（按 val 的固定顺序切片）：在 val[300:1000] 上调，在 val[0:300] 上报告
  python scripts/posthoc.py --run runs/s2_caserag_k5_nothink_val1000 --split val --tune_on 300:1000 --eval_on 0:300

  # 用验证集调好的参数处理测试集（测试集上绝不调参）
  python scripts/posthoc.py --run runs/X_test --split test --params runs/s2_caserag_k5_nothink_val1000_post/posthoc_params.json

输出目录（默认 <run>_post）里有 {split}_preds.jsonl / {split}_metrics.json / {split}_per_case.jsonl，
可以直接交给 evaluate.py --runs / --compare；posthoc_report.md 是本脚本的完整报告。
"""
import argparse
import os
import shutil

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.decision import IDENTITY, KNNVoter, PrecedentIndex, decide, describe, param_grid, tune
from tcmskill.metrics import paired_bootstrap, prf
from tcmskill.prescriber import Prediction
from tcmskill.runner import evaluate_run
from tcmskill.utils import load_json, read_jsonl, save_json, setup_logging, write_jsonl


def parse_slice(s):
    a, b = s.split(":")
    return int(a or 0), (int(b) if b else None)


def subsets_of(items):
    subs = {"overall": [], "dedup": [], "near_dup": [], "prec_unique": [], "prec_ambiguous": [], "prec_none": [], "prev_rare": [], "prev_common": []}
    for i, it in enumerate(items):
        c = it["case"]
        subs["overall"].append(i)
        subs["near_dup" if c.flags.get("near_dup") else "dedup"].append(i)
        subs["prec_" + it["prec"][0]].append(i)
        b = c.flags.get("prev_bucket")
        if f"prev_{b}" in subs:
            subs[f"prev_{b}"].append(i)
    return subs


COLS = [("overall", "F1"), ("dedup", "去重"), ("near_dup", "近重复"), ("prec_unique", "唯一先例"), ("prec_ambiguous", "先例歧义"), ("prec_none", "无先例"), ("prev_rare", "rare")]


def table(items, rows):
    subs = subsets_of(items)
    head = "| 策略 | " + " | ".join(f"{lab}(n={len(subs[k])})" for k, lab in COLS) + " | P | R | 平均味数 |"
    lines = [head, "|" + "---|" * (len(COLS) + 4)]
    for name, herbs_list in rows:
        prfs = np.array([prf(h, it["case"].herbs) for h, it in zip(herbs_list, items)])
        cells = []
        for k, _ in COLS:
            idx = subs[k]
            cells.append(f"{prfs[idx, 2].mean():.3f}" if idx else "—")
        lens = np.mean([len(h) for h in herbs_list])
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {prfs[:, 0].mean():.3f} | {prfs[:, 1].mean():.3f} | {lens:.1f} |")
    return "\n".join(lines)


def apply_params(items, p):
    return [decide(it["case"], it["llm"], it["kv"], it["prec"], p) for it in items]


def crossfit(items, grid, folds, seed=0):
    """Out-of-fold decisions: fold f is processed with parameters tuned on the other folds."""
    rng = np.random.default_rng(seed)
    fold_of = rng.permutation(len(items)) % folds
    out = [None] * len(items)
    chosen = []
    for f in range(folds):
        train = [it for it, g in zip(items, fold_of) if g != f]
        p, _, _ = tune(train, grid)
        chosen.append(p)
        for i, (it, g) in enumerate(zip(items, fold_of)):
            if g == f:
                out[i] = decide(it["case"], it["llm"], it["kv"], it["prec"], p)
    return out, chosen


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--run", required=True, help="含 {split}_preds.jsonl 的 run 目录")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tune_on", default=None, help="调参段，如 300:1000（按 split 的固定顺序）")
    ap.add_argument("--eval_on", default=None, help="报告段，如 0:300（缺省 = 调参段以外的全部病例）")
    ap.add_argument("--params", default=None, help="直接使用已调好的参数文件（posthoc_params.json），不调参")
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--memory_n", type=int, default=0, help="先例 / kNN 记忆：0 = 全部训练集（与 case_rag all_train 一致）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config(args.config, overrides=args.set)
    res = Resources(cfg)

    pred_path = os.path.join(args.run, f"{args.split}_preds.jsonl")
    raw = {r["cid"]: r for r in read_jsonl(pred_path)}
    order = [c for c in res.select(args.split) if c in raw]
    pos = {c: i for i, c in enumerate(res.select(args.split))}
    mem = res.select("train", args.memory_n or None)
    prec_idx, voter = PrecedentIndex(res, mem), KNNVoter(res, mem, args.knn_k)
    cases = res.get(order)
    kvs = voter.votes(cases)
    items = []
    for c, kv in zip(cases, kvs):
        r = raw[c.cid]
        llm = list(r.get("herbs") or []) if r.get("ok", True) else []
        items.append({"case": c, "llm": llm, "kv": kv, "prec": prec_idx.lookup(c)})
    print(f"== {args.run} | {args.split} 共 {len(items)} 例 | 记忆 {len(mem)} 例训练集 | kNN k={args.knn_k}")

    gate_grid = [dict(IDENTITY), {**IDENTITY, "gate": "unique"}]
    fuse_grid = [p for p in param_grid() if p["gate"] == "off"]
    full_grid = param_grid()
    report = [f"# 决策层报告：{args.run}（{args.split}）\n"]

    if args.params:
        P = load_json(args.params)
        final = P["final"]
        ev = items
        decisions = apply_params(ev, final)
        mode = f"固定参数（{args.params}）"
        rows = [("原始 LLM", [it["llm"] for it in ev]), ("决策层（固定参数）", [h for h, _ in decisions])]
        params_out = P
    elif args.tune_on:
        a, b = parse_slice(args.tune_on)
        in_tune = lambda c: a <= pos[c] < (b if b is not None else 10**9)
        tune_items = [it for it in items if in_tune(it["case"].cid)]
        if args.eval_on:
            ea, eb = parse_slice(args.eval_on)
            ev = [it for it in items if ea <= pos[it["case"].cid] < (eb if eb is not None else 10**9) and not in_tune(it["case"].cid)]
        else:
            ev = [it for it in items if not in_tune(it["case"].cid)]
        if not tune_items or not ev:
            raise SystemExit(f"调参段 {len(tune_items)} 例 / 报告段 {len(ev)} 例：run 里的病例不够，请先用更大的 --n_test 跑 LLM")
        pg, _, _ = tune(tune_items, gate_grid)
        pf, _, _ = tune(tune_items, fuse_grid)
        final, f_tune, top = tune(tune_items, full_grid)
        mode = f"调参段 {args.tune_on}（{len(tune_items)} 例）→ 报告段 {args.eval_on or '其余'}（{len(ev)} 例）"
        decisions = apply_params(ev, final)
        rows = [
            ("原始 LLM", [it["llm"] for it in ev]),
            ("+ 唯一先例照抄", [h for h, _ in apply_params(ev, pg)]),
            ("+ kNN 融合/味数校准", [h for h, _ in apply_params(ev, pf)]),
            ("+ 两者（最终）", [h for h, _ in decisions]),
        ]
        params_out = {"final": final, "gate_only": pg, "fuse_only": pf, "tuned_on": args.tune_on, "n_tune": len(tune_items), "f1_tune": f_tune,
                      "top10_on_tune": [{"f1": round(f, 4), **p} for f, p in top]}
        report.append("调参段前 10 名设置：\n" + "\n".join(f"- F1 {f:.4f}  {describe(p)}" for f, p in top))
    else:
        ev = items
        g_out, _ = crossfit(ev, gate_grid, args.folds)
        f_out, _ = crossfit(ev, fuse_grid, args.folds)
        decisions, chosen = crossfit(ev, full_grid, args.folds)
        final, f_all, top = tune(ev, full_grid)
        mode = f"{args.folds} 折交叉拟合（{len(ev)} 例，每例用其余各折调出的参数）"
        rows = [
            ("原始 LLM", [it["llm"] for it in ev]),
            ("+ 唯一先例照抄", [h for h, _ in g_out]),
            ("+ kNN 融合/味数校准", [h for h, _ in f_out]),
            ("+ 两者（最终）", [h for h, _ in decisions]),
        ]
        params_out = {"final": final, "fold_params": chosen, "n_tune": len(ev), "f1_in_sample": f_all,
                      "note": "final = 在全部这些病例上调出的参数，用于处理测试集；报告中的数字来自交叉拟合（样本外）",
                      "top10_in_sample": [{"f1": round(f, 4), **p} for f, p in top]}
        report.append("各折选中的设置：\n" + "\n".join(f"- 第 {i + 1} 折：{describe(p)}" for i, p in enumerate(chosen)))

    no_llm = [decide(it["case"], [], it["kv"], it["prec"], {"gate": "unique"})[0] for it in ev]
    rows.append(("参照：kNN（不用 LLM）", [decide(it["case"], [], it["kv"], ("none", [], []), IDENTITY)[0] for it in ev]))
    rows.append(("参照：先例照抄 + kNN（不用 LLM）", no_llm))
    tab = table(ev, rows)
    how = {}
    for _, h in decisions:
        how[h] = how.get(h, 0) + 1
    print(f"\n模式：{mode}\n最终设置：{describe(final)}\n\n{tab}\n\n决策路径：{how}")
    report += [f"\n模式：{mode}", f"\n最终设置（用于测试集）：{describe(final)}  `{final}`\n", tab, f"\n决策路径：{how}"]

    # ------------------------------------------------------------ paired test: final vs raw LLM
    base_f = [prf(it["llm"], it["case"].herbs)[2] for it in ev]
    fin_f = [prf(h, it["case"].herbs)[2] for (h, _), it in zip(decisions, ev)]
    subs = subsets_of(ev)
    lines = ["\n配对 bootstrap（决策层 − 原始 LLM）："]
    for k in ("overall", "dedup", "prec_unique", "prec_none"):
        idx = subs[k]
        if len(idx) >= 5:
            t = paired_bootstrap([fin_f[i] for i in idx], [base_f[i] for i in idx])
            lines.append(f"  [{k:12s}] ΔF1 {t['diff']:+.4f}  95%CI ({t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f})  p={t['p_value']:.4f}  n={t['n']}")
    print("\n".join(lines))
    report += lines

    # ------------------------------------------------------------ write a run dir that evaluate.py understands
    out = args.out or args.run.rstrip("/") + "_post"
    os.makedirs(out, exist_ok=True)
    preds = []
    for (h, how_i), it in zip(decisions, ev):
        r = raw[it["case"].cid]
        preds.append(Prediction(cid=it["case"].cid, herbs=h, ok=bool(h), json_status=r.get("json_status", "unknown"),
                                finish_reason=r.get("finish_reason"), request_failed=bool(r.get("request_failed", False)),
                                entries=[{"name": x, "role": "", "source": how_i, "targets": []} for x in h]))
    rows_out = []
    for p, (_, how_i), it in zip(preds, decisions, ev):
        d = p.to_dict()
        d["decision"] = how_i
        d["llm_herbs"] = it["llm"]
        d["precedent"] = it["prec"][0]
        rows_out.append(d)
    write_jsonl(os.path.join(out, f"{args.split}_preds.jsonl"), rows_out)
    evaluate_run(res, preds, out, args.split)
    save_json(os.path.join(out, "posthoc_params.json"), params_out)
    save_json(os.path.join(out, "config.json"), {"posthoc_of": args.run, "mode": mode, "final": final, "knn_k": args.knn_k, "memory_n": args.memory_n,
                                                 "base_config": load_json(os.path.join(args.run, "config.json")) if os.path.exists(os.path.join(args.run, "config.json")) else None})
    if os.path.exists(os.path.join(args.run, "usage.json")):
        shutil.copy(os.path.join(args.run, "usage.json"), os.path.join(out, "usage.json"))
    with open(os.path.join(out, "posthoc_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n写入 {out}（posthoc_params.json 里的 final 用于测试集）")


if __name__ == "__main__":
    main()
