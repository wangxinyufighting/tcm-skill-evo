"""评测与对比。

python scripts/evaluate.py --runs runs/knn_nall runs/ours_n2000_random_deepseek-flash   # 汇总表（Markdown）
python scripts/evaluate.py --split val --runs runs/s1_* runs/s2_* --slice 0:300           # 只看 val 前 300 例（与第 1 步同一批）
python scripts/evaluate.py --split val --runs runs/s2_* --common                          # 只比较所有 run 都有的病例
python scripts/evaluate.py --compare runs/ours_n2000_random_deepseek-flash runs/case_rag_n2000_random_deepseek-flash
python scripts/evaluate.py --pred outputs/sft_qwen_test.jsonl --out runs/sft_qwen         # 外部预测：每行 {"cid":..., "herbs":[...]} 或 {"cid":..., "text":"药1、药2"}

"唯一先例" = 训练集里有描述完全相同、且药物完全相同的病例（照抄即可）；"无先例" = 训练集里没有描述完全相同的病例。
这两列把"抄得对"和"真正需要推理"的病例分开，后者才是技能 / 推理方法该拉开差距的地方。
"""
import argparse
import glob
import os
import re

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.decision import PrecedentIndex
from tcmskill.metrics import paired_bootstrap
from tcmskill.prescriber import Prediction
from tcmskill.runner import evaluate_run, format_metrics
from tcmskill.utils import load_json, read_jsonl, setup_logging


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


def parse_slice(s):
    if not s:
        return None
    a, b = s.split(":")
    return int(a or 0), (int(b) if b else None)


class Subsets:
    """cid -> subset membership (flags from data prep + exact-description precedents in the training split)."""

    def __init__(self, res):
        self.res = res
        self.pidx = PrecedentIndex(res, res.splits["train"])
        self._kind = {}

    def kind(self, cid):
        if cid not in self._kind:
            self._kind[cid] = self.pidx.lookup(self.res.cases[cid])[0]
        return self._kind[cid]

    def ids(self, sub, cids):
        fl = lambda c: self.res.cases[c].flags or {}
        if sub == "overall":
            return list(cids)
        if sub == "dedup":
            return [c for c in cids if not fl(c).get("near_dup")]
        if sub == "near_dup":
            return [c for c in cids if fl(c).get("near_dup")]
        if sub.startswith("prev_"):
            return [c for c in cids if fl(c).get("prev_bucket") == sub[5:]]
        if sub.startswith("prec_"):
            return [c for c in cids if self.kind(c) == sub[5:]]
        raise KeyError(sub)


def per_case(d, split):
    p = os.path.join(d, f"{split}_per_case.jsonl")
    return {r["cid"]: r for r in read_jsonl(p)} if os.path.exists(p) else None


def summary(run_dirs, split, res, sl=None, common=False):
    subs = Subsets(res)
    wanted = None
    if sl:
        order = res.select(split)
        wanted = set(order[sl[0] : sl[1]])
    loaded = []
    for d in run_dirs:
        pc = per_case(d, split)
        if pc is None:
            print(f"跳过 {d}（没有 {split}_per_case.jsonl）")
            continue
        if wanted is not None:
            pc = {c: r for c, r in pc.items() if c in wanted}
        cost = None
        if os.path.exists(os.path.join(d, "usage.json")):
            cost = load_json(os.path.join(d, "usage.json"))["total"]["cost_usd"]
        loaded.append((os.path.basename(d.rstrip("/")), pc, cost))
    if not loaded:
        return
    if common:
        keep = set.intersection(*[set(pc) for _, pc, _ in loaded])
        loaded = [(n, {c: r for c, r in pc.items() if c in keep}, cost) for n, pc, cost in loaded]
        print(f"（--common：只用所有 run 共有的 {len(keep)} 例）\n")
    elif len({len(pc) for _, pc, _ in loaded}) > 1:
        print("（注意：各 run 的病例数不同，可加 --slice 或 --common 对齐）\n")
    cols = [("overall", "F1"), ("dedup", "F1 去重"), ("near_dup", "F1 近重复"), ("prec_unique", "F1 唯一先例"), ("prec_none", "F1 无先例"),
            ("prev_rare", "F1 rare"), ("prev_common", "F1 common")]
    head = "| run | n | " + " | ".join(lab for _, lab in cols) + " | soft F1 | P | R | 平均味数 | 味数误差 | 配伍禁忌率 | 费用($) |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    mean = lambda pc, ids, k: float(np.mean([pc[c][k] for c in ids if k in pc[c]])) if ids and any(k in pc[c] for c in ids) else None
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    for name, pc, cost in loaded:
        ids = list(pc)
        cells = [fmt(mean(pc, subs.ids(s, ids), "f1")) for s, _ in cols]
        print(f"| {name} | {len(ids)} | " + " | ".join(cells) + f" | {fmt(mean(pc, ids, 'soft_f1'))} | {fmt(mean(pc, ids, 'p'))} | {fmt(mean(pc, ids, 'r'))} | "
              f"{mean(pc, ids, 'len_pred') or 0:.1f} | {mean(pc, ids, 'len_abs_err') or 0:.2f} | {fmt(mean(pc, ids, 'unsafe'))} | {'' if cost is None else f'{cost:.2f}'} |")
    ids0 = list(loaded[0][1])
    sizes = {lab: len(subs.ids(s, ids0)) for s, lab in cols}
    print(f"\n子集大小（{loaded[0][0]}）：{sizes}" + ("；费用是整个 run 的费用" if sl else ""))
    print("\n| run | 完整 JSON | 修复残片 | 空处方 | 最终截断 | 任一轮截断 | 请求失败 | 失败 | 元数据未知 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, pc, _ in loaded:
        ids = list(pc)
        keys = ("json_complete", "json_repaired_fragment", "empty_prescription", "truncated", "had_truncation", "request_failed", "error", "json_unknown")
        values = []
        for k in keys:
            v = mean(pc, ids, k)
            values.append("未知" if v is None else f"{v:.1%}")
        print(f"| {name} | " + " | ".join(values) + " |")


def compare(a, b, split, res, sl=None):
    subs = Subsets(res)
    A, B = per_case(a, split), per_case(b, split)
    common = sorted(set(A) & set(B))
    if sl:
        order = res.select(split)
        wanted = set(order[sl[0] : sl[1]])
        common = [c for c in common if c in wanted]
    print(f"共同病例 {len(common)} 例：{os.path.basename(a.rstrip('/'))}  vs  {os.path.basename(b.rstrip('/'))}")
    for sub in ("overall", "dedup", "near_dup", "prec_unique", "prec_none", "prev_rare"):
        ids = subs.ids(sub, common)
        if len(ids) < 5:
            continue
        for metric in ("f1", "soft_f1"):
            if metric not in A[ids[0]] or metric not in B[ids[0]]:
                continue
            t = paired_bootstrap([A[c][metric] for c in ids], [B[c][metric] for c in ids])
            print(f"  [{sub:11s}] {metric:7s} 差值 {t['diff']:+.4f}  95%CI ({t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f})  p={t['p_value']:.4f}  n={t['n']}")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--runs", nargs="*", default=None, help="run 目录（可用通配符）")
    ap.add_argument("--compare", nargs=2, default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--slice", default=None, help="只统计 split 固定顺序里的这一段，如 0:300")
    ap.add_argument("--common", action="store_true", help="只统计所有 run 都有的病例")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config(args.config)
    res = Resources(cfg)
    sl = parse_slice(args.slice)
    if args.pred:
        out = args.out or os.path.splitext(args.pred)[0] + "_eval"
        os.makedirs(out, exist_ok=True)
        agg = evaluate_run(res, load_external(args.pred, res), out, args.split)
        print(format_metrics(agg))
        print(f"\n写入 {out}")
    if args.runs is not None:
        dirs = []
        for pat in args.runs or ["runs/*"]:
            dirs += sorted(glob.glob(pat))
        summary([d for d in dirs if os.path.isdir(d)], args.split, res, sl, args.common)
    if args.compare:
        compare(args.compare[0], args.compare[1], args.split, res, sl)


if __name__ == "__main__":
    main()
