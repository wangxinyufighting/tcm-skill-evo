"""评测与对比。

python scripts/evaluate.py --runs runs/knn_nall runs/ours_n2000_random_deepseek-flash   # 汇总表（Markdown）
python scripts/evaluate.py --compare runs/ours_n2000_random_deepseek-flash runs/case_rag_n2000_random_deepseek-flash
python scripts/evaluate.py --pred outputs/sft_qwen_test.jsonl --out runs/sft_qwen         # 外部预测：每行 {"cid":..., "herbs":[...]} 或 {"cid":..., "text":"药1、药2"}
"""
import argparse
import glob
import os
import re

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.metrics import paired_bootstrap
from tcmskill.prescriber import Prediction
from tcmskill.runner import evaluate_run, format_metrics
from tcmskill.utils import load_json, read_jsonl, setup_logging

SUBSETS = ["overall", "dedup", "near_dup", "prev_rare", "prev_mid", "prev_common"]


def load_external(path, res):
    preds = []
    for r in read_jsonl(path):
        if isinstance(r.get("herbs"), list):
            names = r["herbs"]
        else:
            names = re.split(r"[、，,；;\s]+", str(r.get("text", "")))
        herbs, oov = res.lexicon.normalize_list(names)
        preds.append(Prediction(cid=r["cid"], herbs=herbs, oov=oov, ok=bool(herbs)))
    return preds


def summary(run_dirs, split):
    rows = []
    for d in run_dirs:
        p = os.path.join(d, f"{split}_metrics.json")
        if not os.path.exists(p):
            print(f"跳过 {d}（没有 {split}_metrics.json）")
            continue
        m = load_json(p)
        cost = None
        if os.path.exists(os.path.join(d, "usage.json")):
            cost = load_json(os.path.join(d, "usage.json"))["total"]["cost_usd"]
        rows.append((os.path.basename(d.rstrip("/")), m, cost))
    head = "| run | n | F1 | F1 去重 | F1 近重复 | F1 rare | F1 common | soft F1 | P | R | 平均味数 | 味数误差 | 配伍禁忌率 | 费用($) |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    for name, m, cost in rows:
        g = lambda s, k="f1": (f"{m[s][k]:.3f}" if s in m and m[s].get("n") else "—")
        o = m["overall"]
        print(
            f"| {name} | {o['n']} | {g('overall')} | {g('dedup')} | {g('near_dup')} | {g('prev_rare')} | {g('prev_common')} | {o.get('soft_f1', 0):.3f} | {o['p']:.3f} | {o['r']:.3f} | {o['len_pred']:.1f} | {o['len_abs_err']:.2f} | {o.get('unsafe', 0):.3f} | {'' if cost is None else f'{cost:.2f}'} |"
        )
    print("\n| run | 完整 JSON | 修复残片 | 空处方 | 最终截断 | 任一轮截断 | 请求失败 | 失败 | 元数据未知 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, m, _ in rows:
        o = m["overall"]
        keys = ("json_complete", "json_repaired_fragment", "empty_prescription", "truncated", "had_truncation", "request_failed", "error", "json_unknown")
        values = [f"{o[k]:.1%}" if k in o else "未知" for k in keys]
        print(f"| {name} | " + " | ".join(values) + " |")


def compare(a, b, split, res):
    def per_case(d):
        return {r["cid"]: r for r in read_jsonl(os.path.join(d, f"{split}_per_case.jsonl"))}

    A, B = per_case(a), per_case(b)
    common = sorted(set(A) & set(B))
    print(f"共同病例 {len(common)} 例：{os.path.basename(a)}  vs  {os.path.basename(b)}")
    for sub in ("overall", "dedup", "near_dup"):
        if sub == "overall":
            ids = common
        else:
            want = sub == "near_dup"
            ids = [c for c in common if bool(res.cases[c].flags.get("near_dup")) == want]
        if not ids:
            continue
        for metric in ("f1", "soft_f1"):
            if metric not in A[ids[0]]:
                continue
            t = paired_bootstrap([A[c][metric] for c in ids], [B[c][metric] for c in ids])
            print(f"  [{sub:8s}] {metric:7s} 差值 {t['diff']:+.4f}  95%CI ({t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f})  p={t['p_value']:.4f}  n={t['n']}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None, help="run 目录（可用通配符）")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config(args.config)
    if args.pred:
        res = Resources(cfg)
        out = args.out or os.path.splitext(args.pred)[0] + "_eval"
        os.makedirs(out, exist_ok=True)
        agg = evaluate_run(res, load_external(args.pred, res), out, args.split)
        print(format_metrics(agg))
        print(f"\n写入 {out}")
    if args.runs is not None:
        dirs = []
        for pat in args.runs or ["runs/*"]:
            dirs += sorted(glob.glob(pat))
        summary([d for d in dirs if os.path.isdir(d)], args.split)
    if args.compare:
        compare(args.compare[0], args.compare[1], args.split, Resources(cfg))


if __name__ == "__main__":
    main()
