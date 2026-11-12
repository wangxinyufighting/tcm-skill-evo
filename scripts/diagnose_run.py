"""诊断一个 run：技能到底有没有用？技能为什么少？（只读结果文件，不调 API，不碰测试集）

python scripts/diagnose_run.py --run runs/ours_think16k_n500_val300_connfix_v1 --split val

输出：
  1. 按"每味药的来源"统计精度：方证技能(F) / 加减技能(M) / 原则(P) / 自由文本(S) / 参考医案(case) / 经验(note) / 模型自己(self)
  2. 用了方证技能 vs 没用的病例 F1；检索到但没采用的情况
  3. 建库过程的写入统计：创建 / 合并 / 修订 / 各类拒绝原因；治理统计
  4. 技能库概况 + 全部方证技能 + 证据最多的加减规则
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.metrics import prf
from tcmskill.skills import SkillLibrary
from tcmskill.utils import read_jsonl


NAMED = {"base": "底方保留（选方加减）", "add": "加味（选方加减）", "precedent": "先例照抄", "llm": "LLM 原样",
         "fused": "决策层融合", "knn_fallback": "kNN 兜底"}


def src_type(s: str) -> str:
    if s in ("case", "note", "self"):
        return s
    if s in NAMED:
        return NAMED[s]
    return {"F": "F 方证技能", "M": "M 加减技能", "P": "P 原则", "S": "S 自由文本"}.get(s[:1], "其他")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--top_mods", type=int, default=20)
    ap.add_argument("--top_formula", type=int, default=30)
    args = ap.parse_args()
    res = Resources(load_config(args.config))
    pred_path = os.path.join(args.run, f"{args.split}_preds.jsonl")
    preds = [p for p in read_jsonl(pred_path) if p.get("ok", True)]
    print(f"== {args.run} | {args.split} 预测 {len(preds)} 例（ok=true）")

    # ------------------------------------------------------------ 1. precision by source
    hit, tot = Counter(), Counter()
    for p in preds:
        G = set(res.cases[p["cid"]].herbs)
        for e in p.get("entries", []):
            t = src_type(str(e.get("source", "self")))
            tot[t] += 1
            hit[t] += e["name"] in G
    n_all = sum(tot.values()) or 1
    print("\n[1] 每味药按来源统计（精度 = 这类药里命中原方的比例）")
    print("    来源            药味数    占比     精度")
    for t, n in tot.most_common():
        print(f"    {t:12s} {n:8d} {n / n_all:8.1%} {hit[t] / n:8.3f}")

    # ------------------------------------------------------------ 1b. select_edit: which base, how many edits
    se = [p for p in preds if (p.get("meta") or {}).get("style") == "select_edit"]
    if se:
        print(f"\n[1b] 选方加减（{len(se)} 例）：底方位置 / 加减味数 / 与模型自列 herbs 一致的比例")
        by_base = defaultdict(list)
        for p in se:
            by_base[p["meta"].get("base", "none")].append(prf(p.get("herbs", []), res.cases[p["cid"]].herbs)[2])
        for b in sorted(by_base, key=lambda x: (x == "none", x)):
            print(f"    底方 {b:5s} n={len(by_base[b]):4d}  F1={np.mean(by_base[b]):.3f}")
        print(f"    平均减 {np.mean([p['meta'].get('n_remove', 0) for p in se]):.2f} 味、加 {np.mean([p['meta'].get('n_add', 0) for p in se]):.2f} 味；"
              f"一致率 {np.mean([bool(p['meta'].get('consistent')) for p in se]):.1%}")
        base_only = [prf([e["name"] for e in p.get("entries", []) if e.get("source") == "base"] or p.get("herbs", []), res.cases[p["cid"]].herbs)[2] for p in se]
        print(f"    假如只用底方（不加减）F1={np.mean(base_only):.3f}；实际 F1={np.mean([prf(p.get('herbs', []), res.cases[p['cid']].herbs)[2] for p in se]):.3f}")

    # ------------------------------------------------------------ 2. cases with / without skills
    groups = defaultdict(list)
    for p in preds:
        G = res.cases[p["cid"]].herbs
        f1 = prf(p.get("herbs", []), G)[2]
        used = p.get("used_skills", [])
        retrieved = p.get("retrieved", [])
        if any(u.startswith("F") for u in used):
            groups["采用了方证技能"].append(f1)
        elif used:
            groups["只采用加减/原则技能"].append(f1)
        elif retrieved:
            groups["检索到技能但一条没用"].append(f1)
        else:
            groups["没有检索到技能"].append(f1)
    print("\n[2] 按技能使用情况分组的 F1（注意：分组不是随机的，只能看相关性）")
    for k, v in groups.items():
        print(f"    {k:16s} n={len(v):4d}  F1={np.mean(v):.3f}")

    # ------------------------------------------------------------ 3. write / govern statistics
    log_path = os.path.join(args.run, "stream_log.jsonl")
    if os.path.exists(log_path):
        logs = read_jsonl(log_path)
        w, g = Counter(), Counter()
        clusters = kept = 0
        for l in logs:
            ob = l.get("observe", {})
            clusters += ob.get("clusters", 0)
            kept += ob.get("kept", 0)
            w.update(ob.get("write", {}))
            g.update(ob.get("govern", {}) or {})
        print(f"\n[3] 建库 {len(logs)} 个窗口：反思 {kept} 例，聚类 {clusters} 个（= 归纳调用数）")
        created = {k: v for k, v in w.items() if k.startswith("created")}
        rejected = {k: v for k, v in w.items() if "rejected" in k or k.startswith("invalid")}
        other = {k: v for k, v in w.items() if k not in created and k not in rejected}
        print(f"    创建：{created}")
        print(f"    拒绝：{dict(sorted(rejected.items(), key=lambda x: -x[1]))}")
        print(f"    其他：{other}")
        print(f"    治理：{dict(g)}")
        print("    逐窗 prequential F1：" + " ".join(f"{l['f1']:.2f}" for l in logs if "f1" in l))

    # ------------------------------------------------------------ 4. library
    mem_path = next((os.path.join(args.run, n) for n in ("memory_final.json", "memory.json") if os.path.exists(os.path.join(args.run, n))), None)
    if mem_path:
        mem = json.load(open(mem_path, encoding="utf-8"))
        lib = SkillLibrary()
        lib.load_state_dict(mem.get("library", {}))
        act = lib.active()
        print(f"\n[4] 技能库（{os.path.basename(mem_path)}）：{lib.stats()}；已见病例 {len(mem.get('store', {}).get('cids', []))}")
        formulas = sorted([s for s in act if s.branch == "formula"], key=lambda s: (-s.n_used, -s.utility))
        if len(formulas) > args.top_formula:
            print(f"    —— 方证技能共 {len(formulas)} 条，只列采用最多的 {args.top_formula} 条（--top_formula 调整）")
        for s in formulas[: args.top_formula]:
            print(f"    {s.render(max_mods=8)}\n      效用 {s.utility:.2f} 采用 {s.n_used} 检索 {s.n_ret} 检验 {s.vstats}")
        mods = sorted([s for s in act if s.branch == "modification"], key=lambda s: -(s.vstats or {}).get("n_xy", 0))
        print(f"    —— 证据最多的 {args.top_mods} 条加减规则（共 {len(mods)} 条）")
        for s in mods[: args.top_mods]:
            print(f"    {s.render()}  效用 {s.utility:.2f} 采用 {s.n_used}")
        for s in [s for s in act if s.branch == "principle"][:8]:
            print(f"    {s.render()}  效用 {s.utility:.2f} 采用 {s.n_used}")


if __name__ == "__main__":
    main()
