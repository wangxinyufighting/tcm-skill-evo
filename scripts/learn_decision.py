"""学习式决策层（零 API 费用）：从反馈里学"哪些药该留"——LLM 和检索负责提议，经验负责取舍。

  # A. val 上 5 折交叉拟合 + 消融：不用 LLM / 加各个 LLM run / 再加技能特征
  python scripts/learn_decision.py --split val \\
      --runs runs/s2_caserag_k5_nothink_val1000 runs/s2_select_k5_nothink_val1000 \\
      --skills runs/offline_v1/memory_final.json --out runs/s3_learned_val1000

  # B. 练习协议：在"训练病例上的练习作答"里学，在 val 上评测（--fit_runs 与 --runs 按顺序一一对应，须是同一种方法）
  python scripts/learn_decision.py --split val --runs runs/s2_caserag_k5_nothink_val1000 \\
      --fit_split train --fit_runs runs/s3_caserag_k5_nothink_train2000 --curve 250,500,1000,2000 --out runs/s3_practice_caserag

  # B'. 不用 LLM 的练习（零费用）：在训练流前 N 例上学（检索时排除病例自己），在 val 上评测 —— 强的非 LLM 基线
  python scripts/learn_decision.py --split val --fit_split train --fit_runs --fit_n 10000 \
      --curve 500,1000,2000,5000,10000 --out runs/s3_learned_nollm_practice

  # C. 把保存的模型用到测试集（测试集上不训练、不调参）
  python scripts/learn_decision.py --split test --runs runs/X_test --model runs/s3_practice_caserag/model.pkl --out runs/X_test_learned

输出目录：{split}_preds.jsonl / {split}_metrics.json / {split}_per_case.jsonl（evaluate.py 可直接读）、
learned_report.md（消融表 + 配对检验 + 学到的权重）、model.pkl（最终设置在全部训练数据上拟合，用于测试集）。
"""
import argparse
import os
import pickle

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.decision import fuse
from tcmskill.learned import FeatureBuilder, apply, crossfit, fit, lr_weights
from tcmskill.metrics import paired_bootstrap, prf
from tcmskill.prescriber import Prediction
from tcmskill.runner import evaluate_run
from tcmskill.utils import load_json, read_jsonl, save_json, setup_logging, write_jsonl

COLS = [("all", "F1"), ("dedup", "去重"), ("near_dup", "近重复"), ("prec_unique", "唯一先例"), ("prec_ambiguous", "先例歧义"),
        ("prec_none", "无先例"), ("prev_rare", "rare")]
FEAT_ZH = {"knn_vote": "kNN 投票", "top5_share": "前 5 例中出现比例", "in_top1": "在最相似那例中", "max_sim_top5": "用它的近邻的最高相似度",
           "log_prior": "全库频次(log)", "sim1": "最高相似度(病例级)", "knn_len": "kNN 味数(病例级)", "prec_herb": "唯一先例含此药",
           "prec_unique": "有唯一先例", "prec_ambiguous": "先例歧义", "rule_loglift": "加减规则 lift(log)", "rule_conf": "加减规则置信度",
           "fam_core_share": "近邻方族核心药比例"}


def load_run(d, split):
    path = os.path.join(d, f"{split}_preds.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}")
    return {r["cid"]: (list(r.get("herbs") or []) if r.get("ok", True) else []) for r in read_jsonl(path)}


def subsets(cases, kinds):
    s = {k: [] for k, _ in COLS}
    for i, (c, k) in enumerate(zip(cases, kinds)):
        fl = c.flags or {}
        s["all"].append(i)
        s["near_dup" if fl.get("near_dup") else "dedup"].append(i)
        s["prec_" + k].append(i)
        if fl.get("prev_bucket") == "rare":
            s["prev_rare"].append(i)
    return s


def table(cases, kinds, rows):
    S = subsets(cases, kinds)
    lines = ["| 预测 | " + " | ".join(f"{lab}(n={len(S[k])})" for k, lab in COLS) + " | P | R | 平均味数 |", "|" + "---|" * (len(COLS) + 4)]
    for name, preds in rows:
        prfs = np.array([prf(h, c.herbs) for h, c in zip(preds, cases)])
        cells = [f"{prfs[S[k], 2].mean():.3f}" if S[k] else "—" for k, _ in COLS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {prfs[:, 0].mean():.3f} | {prfs[:, 1].mean():.3f} | {np.mean([len(h) for h in preds]):.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--runs", nargs="*", default=[], help="LLM run 目录（作为特征来源；可以不给 = 只用检索特征）")
    ap.add_argument("--n", type=int, default=1000, help="不给 --runs 时评测 split 的前 N 例")
    ap.add_argument("--skills", default=None, help="技能库 memory_final.json（加技能特征）")
    ap.add_argument("--fit_split", default=None, choices=["train", "val"])
    ap.add_argument("--fit_runs", nargs="*", default=None, help="练习作答的 run（与 --runs 一一对应）；只写 --fit_runs 不跟目录 = 不用 LLM 的练习（零费用）")
    ap.add_argument("--fit_n", type=int, default=5000, help="不用 LLM 的练习：取训练流前 N 例")
    ap.add_argument("--curve", default=None, help="练习协议下的经验曲线，如 250,500,1000,2000")
    ap.add_argument("--model", default=None, help="直接套用保存的 model.pkl")
    ap.add_argument("--model_kind", default="gb", choices=["gb", "lr"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--knn_k", type=int, default=20)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    setup_logging("WARNING")
    cfg = load_config(args.config, overrides=args.set)
    res = Resources(cfg)
    mem = res.select("train")
    skill_mem = load_json(args.skills) if args.skills else None
    fb = FeatureBuilder(res, mem, args.knn_k, skill_mem, cfg)

    def gather(split, run_dirs, n=None):
        runs = [load_run(d, split) for d in run_dirs]
        if runs:
            common = set.intersection(*[set(r) for r in runs])
            ids = [c for c in res.select(split) if c in common]
        else:
            ids = res.select(split, n)
        return res.get(ids), runs

    cases, runs = gather(args.split, args.runs, args.n)
    names = [os.path.basename(d.rstrip("/")) for d in args.runs]
    print(f"== {args.split}：{len(cases)} 例；LLM run：{names or '无'}；技能特征：{'有' if skill_mem else '无'}；记忆 {len(mem)} 例训练集")
    T = fb.build(cases, runs)
    n_llm = len(runs)

    # ------------------------------------------------------------ configurations (ablation ladder)
    configs = [("检索+先例（不用 LLM）", ["retr"])]
    if n_llm > 1:
        configs += [(f"+ LLM：{names[j]}", ["retr", f"llm{j}"]) for j in range(n_llm)]
    if n_llm:
        configs.append(("+ LLM：全部 run" if n_llm > 1 else f"+ LLM：{names[0]}", ["retr"] + [f"llm{j}" for j in range(n_llm)]))
    if skill_mem:
        configs.append(("+ 技能特征", configs[-1][1] + ["skill"]))
    final_label, final_groups = configs[-1]
    report = [f"# 学习式决策层：{args.out}\n", f"- 评测：{args.split} {len(cases)} 例；LLM run：{names or '无'}；技能：{args.skills or '无'}；模型：{args.model_kind}"]

    preds_by_cfg = {}
    model_obj = None
    if args.model:
        with open(args.model, "rb") as f:
            model_obj = pickle.load(f)
        if model_obj["n_llm"] != n_llm or model_obj["skills"] != bool(skill_mem):
            raise SystemExit(f"模型需要 {model_obj['n_llm']} 个 LLM run、技能特征={model_obj['skills']}，与本次输入不一致")
        final_label, final_groups = model_obj["label"], model_obj["groups"]
        preds_by_cfg[final_label] = apply(model_obj["model"], T, final_groups)
        mode = f"套用 {args.model}"
    elif args.fit_runs is not None:
        if len(args.fit_runs) != n_llm:
            raise SystemExit("--fit_runs 必须与 --runs 一一对应（同样个数、同样顺序、同一种方法）")
        fcases, fruns = gather(args.fit_split or "train", args.fit_runs, args.fit_n)
        if skill_mem is not None:
            lib_cids = set(skill_mem.get("store", {}).get("cids", []))
            overlap = sum(1 for c in fcases if c.cid in lib_cids)
            if overlap:
                print(f"！警告：{overlap} 个练习病例参与了技能库构建，技能特征会泄漏标签。请用 build_skills_offline.py --skip_first {len(fcases)} 重新建库")
        FT = fb.build(fcases, fruns)
        mode = f"练习协议：在 {args.fit_split or 'train'} 的 {len(fcases)} 例练习作答上拟合 → 在 {args.split} 上评测"
        for label, groups in configs:
            preds_by_cfg[label] = apply(fit(FT, groups, args.model_kind), T, groups)
        model_obj = {"model": fit(FT, final_groups, args.model_kind), "groups": final_groups, "label": final_label}
        if args.curve:
            pts = [int(x) for x in args.curve.split(",") if x.strip()]
            report.append("\n## 经验曲线（练习病例数 → val 上的 F1，最终设置）\n\n| 练习病例数 | F1 | 无先例 | 唯一先例 |\n|---|---|---|---|")
            S = subsets(cases, T.kinds)
            for n in pts:
                n = min(n, len(fcases))
                pr = apply(fit(FT, final_groups, args.model_kind, case_subset=range(n)), T, final_groups)
                f = np.array([prf(h, c.herbs)[2] for h, c in zip(pr, cases)])
                report.append(f"| {n} | {f.mean():.3f} | {f[S['prec_none']].mean():.3f} | {f[S['prec_unique']].mean():.3f} |")
                print(f"  经验曲线：练习 {n} 例 → F1 {f.mean():.3f}（无先例 {f[S['prec_none']].mean():.3f}）")
    else:
        mode = f"{args.folds} 折交叉拟合（每例用其余各折训练的模型）"
        for label, groups in configs:
            preds_by_cfg[label] = crossfit(T, groups, args.model_kind, args.folds)
        model_obj = {"model": fit(T, final_groups, args.model_kind), "groups": final_groups, "label": final_label}

    # ------------------------------------------------------------ report
    kv_votes = fb.voter.votes(cases)
    knn = [fuse([], kv, a=0.0, length="knn", add_thr=0.0) for kv in kv_votes]
    gate_knn = [fb.pidx.lookup(c)[1] if k == "unique" else h for c, k, h in zip(cases, T.kinds, knn)]
    rows = [(f"原始 LLM：{n}", [r.get(c.cid, []) for c in cases]) for n, r in zip(names, runs)]
    rows += [("参照：kNN", knn), ("参照：唯一先例照抄 + kNN", gate_knn)]
    rows += [(f"学习：{label}", p) for label, p in preds_by_cfg.items()]
    tab = table(cases, T.kinds, rows)
    print(f"\n模式：{mode}\n\n{tab}")
    report += [f"- 模式：{mode}", f"- 最终设置：{final_label}\n", tab]

    S = subsets(cases, T.kinds)
    fin = preds_by_cfg[final_label]
    fin_f = np.array([prf(h, c.herbs)[2] for h, c in zip(fin, cases)])
    comps = []
    if runs:
        comps.append((f"最终 − 原始 LLM（{names[0]}）", [prf(r, c.herbs)[2] for r, c in zip(rows[0][1], cases)]))
    if "检索+先例（不用 LLM）" in preds_by_cfg and final_label != "检索+先例（不用 LLM）":
        comps.append(("最终 − 不用 LLM 的同一模型（= LLM 的边际价值）", [prf(h, c.herbs)[2] for h, c in zip(preds_by_cfg["检索+先例（不用 LLM）"], cases)]))
    if skill_mem and len(configs) >= 2 and configs[-2][0] in preds_by_cfg:
        comps.append(("最终 − 去掉技能特征（= 技能的边际价值）", [prf(h, c.herbs)[2] for h, c in zip(preds_by_cfg[configs[-2][0]], cases)]))
    lines = ["\n## 配对 bootstrap\n"]
    for title, other in comps:
        lines.append(f"**{title}**")
        for k in ("all", "dedup", "prec_unique", "prec_none", "prev_rare"):
            idx = S[k]
            if len(idx) >= 5:
                t = paired_bootstrap(fin_f[idx].tolist(), [other[i] for i in idx])
                lines.append(f"- [{k}] ΔF1 {t['diff']:+.4f}  95%CI ({t['ci95'][0]:+.4f}, {t['ci95'][1]:+.4f})  p={t['p_value']:.4f}  n={t['n']}")
    print("\n".join(lines))
    report += lines

    if not args.model:
        try:  # readable summary of the learned decision rule
            src = FT if args.fit_runs is not None else T
            w = lr_weights(src, final_groups)
            report.append("\n## 学到了什么（逻辑回归标准化权重，前 12；正 = 更该保留）\n")
            report += [f"- {FEAT_ZH.get(n, n)}：{v:+.2f}" for n, v in w[:12]]
        except Exception as e:  # pragma: no cover
            report.append(f"\n（权重摘要失败：{e}）")

    # ------------------------------------------------------------ outputs
    os.makedirs(args.out, exist_ok=True)
    preds = [Prediction(cid=c.cid, herbs=h, ok=bool(h), json_status="unknown",
                        entries=[{"name": x, "role": "", "source": "learned", "targets": []} for x in h]) for c, h in zip(cases, fin)]
    write_jsonl(os.path.join(args.out, f"{args.split}_preds.jsonl"), [p.to_dict() for p in preds])
    evaluate_run(res, preds, args.out, args.split)
    if model_obj is not None and not args.model:
        with open(os.path.join(args.out, "model.pkl"), "wb") as f:
            pickle.dump({**model_obj, "n_llm": n_llm, "skills": bool(skill_mem), "kind": args.model_kind, "knn_k": args.knn_k,
                         "names": T.names, "fit_runs": args.fit_runs, "runs": names}, f)
    save_json(os.path.join(args.out, "config.json"), {"mode": mode, "runs": args.runs, "fit_runs": args.fit_runs, "skills": args.skills,
                                                      "final": final_label, "groups": final_groups, "model_kind": args.model_kind, "knn_k": args.knn_k})
    costs = [load_json(os.path.join(d, "usage.json"))["total"]["cost_usd"] for d in args.runs if os.path.exists(os.path.join(d, "usage.json"))]
    if costs:  # the decision layer itself is free; the cost is that of the LLM runs it uses
        save_json(os.path.join(args.out, "usage.json"), {"total": {"cost_usd": round(float(sum(costs)), 4)}, "note": "sum over --runs"})
    with open(os.path.join(args.out, "learned_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print(f"\n写入 {args.out}（learned_report.md；model.pkl 用于测试集）")


if __name__ == "__main__":
    main()
