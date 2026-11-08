"""导出专家盲评表：同一病例下，原方与各方法的处方随机打乱、匿名呈现。

python scripts/export_blind_review.py --runs runs/ours_n2000_random_deepseek-flash runs/case_rag_n2000_random_deepseek-flash --n 100 --out review/blind_100
生成：
  <out>.csv       给专家：病例描述 + 处方A/B/C（不标来源）+ 评分列
  <out>_key.csv   答案：每个病例中 A/B/C 分别是哪个方法（评完再揭盲）
"""
import argparse
import csv
import os
import random

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.utils import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--split", default="test")
    ap.add_argument("--no_gt", action="store_true", help="不把原方混进去")
    ap.add_argument("--dedup_only", action="store_true", help="只抽去重子集（与训练集无近重复）")
    ap.add_argument("--out", default="review/blind")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    res = Resources(load_config())
    preds = {}
    for d in args.runs:
        preds[os.path.basename(d.rstrip("/"))] = {r["cid"]: r["herbs"] for r in read_jsonl(os.path.join(d, f"{args.split}_preds.jsonl"))}
    common = set.intersection(*[set(p) for p in preds.values()])
    if args.dedup_only:
        common = {c for c in common if not res.cases[c].flags.get("near_dup")}
    rng = random.Random(args.seed)
    cids = rng.sample(sorted(common), min(args.n, len(common)))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    letters = "ABCDEFG"
    with open(args.out + ".csv", "w", encoding="utf-8-sig", newline="") as f, open(args.out + "_key.csv", "w", encoding="utf-8-sig", newline="") as g:
        w, k = csv.writer(f), csv.writer(g)
        n_opts = len(preds) + (0 if args.no_gt else 1)
        head = ["编号", "主治描述"]
        for i in range(n_opts):
            L = letters[i]
            head += [f"处方{L}", f"{L}_合理性(1-5)", f"{L}_安全性(1-5)", f"{L}_备注"]
        w.writerow(head + ["最佳处方(字母)"])
        k.writerow(["编号", "cid"] + [f"处方{letters[i]}" for i in range(n_opts)])
        for idx, cid in enumerate(cids, 1):
            opts = [(name, p[cid]) for name, p in preds.items()]
            if not args.no_gt:
                opts.append(("原方(GT)", res.cases[cid].herbs))
            rng.shuffle(opts)
            row = [idx, res.cases[cid].desc]
            for _, herbs in opts:
                row += ["、".join(herbs), "", "", ""]
            w.writerow(row + [""])
            k.writerow([idx, cid] + [name for name, _ in opts])
    print(f"写入 {args.out}.csv（给专家）和 {args.out}_key.csv（揭盲用）")


if __name__ == "__main__":
    main()
