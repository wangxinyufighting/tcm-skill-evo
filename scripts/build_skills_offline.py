"""离线批量建技能库：一次读完整个训练集（不走数据流），统计部分不调 LLM。

  # 1) 纯统计（几十秒，零费用）：方族 → 方证技能；全局 症状→药 加减规则；检验；符号执行对比 kNN
  python scripts/build_skills_offline.py --run_name offline_v1

  # 2) 可选：让 LLM 给每个方族起方名、写病机/治法、标君臣佐使（约 1000 次调用，非思考约 $1；触发词与药物不变）
  python scripts/build_skills_offline.py --run_name offline_v1_named --annotate --set llm.thinking=false

  # 3) 用这份技能库测试（不再建库）：技能 + 5 个相似医案
  python scripts/run_method.py --method ours_hybrid --phase test --test_split val --n_test 1000 \\
      --load_memory runs/offline_v1/memory_final.json --set skills.hybrid_cases=5 --set llm.thinking=false \\
      --run_name s2_hybrid_offline_k5_nothink_val1000

输出：runs/<run_name>/memory_final.json（--load_memory 用）、offline_report.md、build_stats.json
"""
import argparse
import os
import time

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.baselines import KNNBaseline
from tcmskill.config import load_config, to_plain
from tcmskill.data import Resources
from tcmskill.offline import OfflineBuilder, symbolic_predict, symbolic_report
from tcmskill.utils import save_json, setup_logging


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--run_name", default="offline_v1")
    ap.add_argument("--n_train", type=int, default=0, help="0 = 全部训练集；>0 = 训练流前 N 例（与在线建库同一批，便于对比）")
    ap.add_argument("--skip_first", type=int, default=0, help="跳过训练流前 N 例（练习协议：练习病例不能参与建库，否则技能特征泄漏标签）")
    ap.add_argument("--annotate", action="store_true", help="调用 LLM 给方族命名（病机/治法/君臣佐使）")
    ap.add_argument("--llm", default=None, choices=["deepseek", "mock", "openai_compatible"])
    ap.add_argument("--eval_n", type=int, default=1000, help="符号执行评测用 val 前 N 例（只读 val，不碰 test）")
    args = ap.parse_args()
    overrides = list(args.set) + ([f"llm.provider={args.llm}"] if args.llm else [])
    cfg = load_config(args.config, overrides=overrides)
    run_dir = os.path.join(cfg.paths.runs_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(run_dir, "run.log"))
    save_json(os.path.join(run_dir, "config.json"), {**to_plain(cfg), "args": vars(args)})
    res = Resources(cfg)
    train = res.select("train")[args.skip_first:]
    if args.n_train:
        train = train[: args.n_train]
    t0 = time.time()
    b = OfflineBuilder(res, cfg, train)
    fams = b.build_formulas()
    b.build_rules()
    llm = None
    if args.annotate:
        from tcmskill.llm import build_llm

        if cfg.llm.provider == "mock":
            cfg.llm.cache_path = ""
        llm = build_llm(cfg)
        llm.call_log_path = os.path.join(run_dir, "llm_calls.jsonl")
        b.annotate(llm)
    secs = time.time() - t0

    lib = b.lib
    fs, ms = lib.active("formula"), lib.active("modification")
    vs = lambda k: float(np.mean([s.vstats.get(k, 0) for s in fs])) if fs else 0.0
    lines = [f"# 离线技能库：{args.run_name}\n",
             f"- 训练病例 {b.N}；症状短语词表 {len(b.vocab)}；方族 {len(fams)} 个（覆盖 {sum(len(f) for f in fams)} 例）",
             f"- 方证技能 {len(fs)}（多触发 {sum(1 for s in fs if len(s.triggers) > 1)}，带加减 {sum(1 for s in fs if s.mods)}）；加减规则 {len(ms)}",
             f"- 方证检验均值：命中 {vs('n_match'):.0f} 例，核心药精度 {vs('prec'):.3f}，族外精度 {vs('prec_out'):.3f}",
             f"- 写入/拒绝统计：{dict(b.stats)}",
             f"- 用时 {secs:.0f}s" + (f"；LLM 命名费用 ${llm.usage.summary()['total']['cost_usd']:.3f}" if llm else "")]

    cases = res.get(res.select("val", args.eval_n))
    sym = symbolic_predict(lib, b.matcher, res, cases)
    knn = [p.herbs for p in KNNBaseline(res, k=int(cfg.knn.k)).fit(train).predict(cases)]
    lines += [f"\n## 符号执行（不调 LLM）vs kNN：val 前 {len(cases)} 例\n", symbolic_report(res, cases, sym, knn),
              "\n说明：技能库单独开方远弱于 kNN 是预期的（核心药是同族方的交集）。它的作用是给 LLM 提供方族与加减线索，"
              "真正的检验是 run_method.py 的 ours_hybrid / ours_select 是否在去重、rare、无先例子集上超过 case_rag。"]
    lines.append("\n## 证据最多的方证技能（前 15）\n")
    lines += [f"- {s.render()}  `{s.vstats}`" for s in sorted(fs, key=lambda s: -s.support)[:15]]
    lines.append("\n## 证据最多的加减规则（前 20）\n")
    lines += [f"- {s.render()}  `conf={s.vstats.get('conf')} lift={s.vstats.get('lift')}`" for s in sorted(ms, key=lambda s: -s.vstats.get("n_xy", 0))[:20]]
    rng = np.random.default_rng(0)
    if fs:
        lines.append("\n## 随机抽样的方证技能（10 条）\n")
        lines += [f"- {fs[i].render()}  `{fs[i].vstats}`" for i in rng.choice(len(fs), size=min(10, len(fs)), replace=False)]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(run_dir, "offline_report.md"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    save_json(os.path.join(run_dir, "memory_final.json"), b.memory(), indent=None)
    save_json(os.path.join(run_dir, "build_stats.json"), {"n_train": b.N, "families": len(fams), "formula": len(fs), "modification": len(ms),
                                                         "stats": dict(b.stats), "secs": round(secs, 1)})
    if llm is not None:
        save_json(os.path.join(run_dir, "usage.json"), llm.usage.summary())
    print(f"\n结果目录：{run_dir}（--load_memory {os.path.join(run_dir, 'memory_final.json')}）")


if __name__ == "__main__":
    main()
