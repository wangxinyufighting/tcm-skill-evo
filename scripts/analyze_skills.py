"""技能库分析。

python scripts/analyze_skills.py --run runs/ours_n2000_random_deepseek-flash
python scripts/analyze_skills.py --run runs/ours_... --formulas kb/formulas.csv     # 教材方剂重发现率（需要你提供方剂表）

输出到 <run>/analysis/：
  library_stats.json    规模、压缩率、效用/支持度分布、多触发（异病同治）与拆分（同病异治）次数
  skills.md             可读的技能库全文（按分支、按效用排序）
  skills_review.csv     抽样技能 + 空白评分列（给中医专家盲评用）
  symbolic_test_*.json  不经 LLM、只做符号执行时技能库在测试集上的覆盖率与 F1
  rediscovery.csv       （可选）教材方剂 vs 学到的方证技能
"""
import argparse
import csv
import json
import os
import random
import re
from collections import Counter

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.metrics import evaluate_predictions, default_subsets
from tcmskill.skills import SkillLibrary, SkillMatcher
from tcmskill.stats import SeenStore
from tcmskill.utils import jaccard, load_json, save_json, setup_logging


def q(xs):
    if not xs:
        return {}
    a = np.asarray(xs, float)
    return {"mean": round(float(a.mean()), 3), "p10": round(float(np.quantile(a, 0.1)), 3), "p50": round(float(np.quantile(a, 0.5)), 3), "p90": round(float(np.quantile(a, 0.9)), 3)}


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--run", required=True)
    ap.add_argument("--memory", default="memory_final.json")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--formulas", default=None, help="CSV：方名,组成[,主治]；组成用、或，分隔")
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    setup_logging("WARNING")
    run_cfg = load_json(os.path.join(args.run, "config.json"))
    cfg = load_config(args.config, method=run_cfg.get("method", {}).get("name"))
    res = Resources(cfg)
    mem = load_json(os.path.join(args.run, args.memory))
    lib = SkillLibrary()
    lib.load_state_dict(mem.get("library", {}))
    store = SeenStore(res)
    store.load_state_dict(mem.get("store", {}))
    matcher = SkillMatcher(store, cfg.skills.match)
    out = os.path.join(args.run, "analysis")
    os.makedirs(out, exist_ok=True)

    # ---------------------------------------------------------------- stats
    act = lib.active()
    by_b = Counter(s.branch for s in act)
    hist = Counter()
    for s in lib.skills.values():
        for h in s.history:
            if "split" in h:
                hist["split"] += 1
            if "异病同治" in h:
                hist["merge_multi_trigger"] += 1
            if "merge" in h:
                hist["merge"] += 1
    stats = {
        "seen_cases": store.n,
        "active_by_branch": dict(by_b),
        "all_by_branch_status": lib.stats(),
        "compression_cases_per_skill": round(store.n / max(len(act), 1), 2),
        "multi_trigger_formula": sum(1 for s in act if s.branch == "formula" and len(s.triggers) > 1),
        "history_events": dict(hist),
        "utility": q([s.utility for s in act]),
        "support": q([s.support for s in act if s.branch in ("formula", "modification")]),
        "core_size": q([len(s.core) for s in act if s.branch == "formula"]),
        "n_used": q([s.n_used for s in act]),
    }
    save_json(os.path.join(out, "library_stats.json"), stats)

    # ---------------------------------------------------------------- readable dump
    lines = [f"# 技能库（{args.run}）", "", f"已见病例 {store.n}；活跃技能 {dict(by_b)}；压缩率 {stats['compression_cases_per_skill']} 例/技能", ""]
    for br, title in (("formula", "方证技能"), ("modification", "药症技能（随症加减）"), ("principle", "原则技能"), ("free", "自由文本技能")):
        pool = sorted([s for s in act if s.branch == br], key=lambda s: (-s.utility, -s.support))
        if not pool:
            continue
        lines += [f"## {title}（{len(pool)}）", ""]
        for s in pool:
            lines.append(f"- {s.render(max_mods=12)}")
            lines.append(f"  - 效用 {s.utility:.2f}｜采用 {s.n_used} 次｜检索 {s.n_ret} 次｜支持 {s.support}｜检验 {json.dumps(s.vstats, ensure_ascii=False)}｜状态 {s.status}")
        lines.append("")
    with open(os.path.join(out, "skills.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---------------------------------------------------------------- expert review sheet
    rng = random.Random(0)
    sample = rng.sample(act, min(args.sample, len(act)))
    with open(os.path.join(out, "skills_review.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "分支", "名称", "适用（触发）", "病机", "治法", "基础方/加药", "随症加减", "原则/内容", "证据例数", "专家_正确性(1-5)", "专家_临床可用性(1-5)", "专家_备注"])
        for s in sample:
            trig = " 或 ".join(t.render() for t in s.triggers)
            herbs = "、".join(f"{c['name']}{('(' + c['role'] + ')') if c.get('role') else ''}" for c in s.core) if s.branch == "formula" else "、".join(s.add)
            mods = "；".join(f"若{'、'.join(m['if'])}加{'、'.join(m.get('add', []))}" for m in s.mods)
            w.writerow([s.id, s.branch, s.name, trig, s.pathogenesis, s.principle, herbs, mods, s.content, s.support, "", "", ""])

    # ---------------------------------------------------------------- symbolic execution on test (no LLM)
    test_ids = res.select("test", args.n_test or None)
    preds, covered = {}, 0
    formulas = lib.active("formula")
    mods = lib.active("modification")
    for cid in test_ids:
        c = res.cases[cid]
        text, sy = res.text(c), set(c.symptoms)
        best, best_sc = None, 0.0
        for s in formulas:
            m = matcher.match(s, text, sy)
            if m >= matcher.fire and m * (0.5 + s.utility) > best_sc:
                best, best_sc = s, m * (0.5 + s.utility)
        herbs = matcher.execute(best, text, sy) if best else []
        fired = sorted([s for s in mods if matcher.match(s, text, sy) >= matcher.fire], key=lambda s: -s.utility)[:5]
        for s in fired:
            herbs += [h for h in s.add if h not in herbs]
        covered += bool(best)
        preds[cid] = herbs
    ev = evaluate_predictions(preds, res.cases, res.herb_info, res.incompat, default_subsets(res.cases, preds))
    cov_ids = [c for c in test_ids if preds[c]]
    symb = {"coverage_formula": round(covered / max(len(test_ids), 1), 4), "coverage_any": round(len(cov_ids) / max(len(test_ids), 1), 4), "metrics": ev["aggregate"]}
    if cov_ids:
        symb["f1_on_covered"] = round(float(np.mean([ev["per_case"][c]["f1"] for c in cov_ids])), 4)
    save_json(os.path.join(out, f"symbolic_test_n{len(test_ids)}.json"), symb)

    # ---------------------------------------------------------------- rediscovery (optional)
    if args.formulas and os.path.exists(args.formulas):
        rows = []
        with open(args.formulas, encoding="utf-8-sig") as f:
            for r in csv.reader(f):
                if not r or r[0] in ("方名", "name"):
                    continue
                comp, _ = res.lexicon.normalize_list(re.split(r"[、，,；;\s]+", r[1]))
                if comp:
                    rows.append((r[0], comp, r[2] if len(r) > 2 else ""))
        rep = []
        for name, comp, ind in rows:
            best = max(formulas, key=lambda s: jaccard(comp, s.core_herbs()), default=None)
            j = jaccard(comp, best.core_herbs()) if best else 0.0
            rep.append({"方名": name, "组成": "、".join(comp), "最相近技能": best.id if best else "", "技能名称": best.name if best else "", "技能基础方": "、".join(best.core_herbs()) if best else "", "Jaccard": round(j, 3)})
        with open(os.path.join(out, "rediscovery.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rep[0].keys()) if rep else ["方名"])
            w.writeheader()
            w.writerows(rep)
        for thr in (0.5, 0.6, 0.7):
            stats[f"rediscovered@{thr}"] = sum(1 for r in rep if r["Jaccard"] >= thr)
        stats["n_textbook_formulas"] = len(rep)
        save_json(os.path.join(out, "library_stats.json"), stats)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in symb.items() if k != "metrics"} | {"symbolic_f1": symb["metrics"]["overall"].get("f1")}, ensure_ascii=False))
    print(f"\n输出：{out}/")


if __name__ == "__main__":
    main()
