"""主实验入口：建库（流式学习）+ 冻结测试。

例：
  # 冒烟测试（不调 API）
  python scripts/run_method.py --method ours --llm mock --n_train 300 --n_test 100
  # 正式 pilot
  python scripts/run_method.py --method ours --n_train 2000 --n_test 500
  # 只测试（用已建好的技能库）
  python scripts/run_method.py --method ours --n_train 2000 --n_test 500 --phase test
  # 跨模型迁移：用 v4-pro 建的库，交给 flash 用
  python scripts/run_method.py --method ours --model deepseek-flash --load_memory runs/ours_n2000_random_deepseek-v4-pro/memory_final.json --run_name transfer_pro2flash
"""
import argparse
import logging
import os

import _bootstrap  # noqa: F401

from tcmskill.config import load_config, to_plain
from tcmskill.data import Resources
from tcmskill.llm import OUTPUT_PROTOCOL_VERSION, build_llm
from tcmskill.methods import build_method
from tcmskill.runner import check_stream_resume, format_metrics, format_output_stats, load_method, run_stream, run_test
from tcmskill.utils import load_json, save_json, setup_logging

LOG = logging.getLogger("run_method")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("--method", required=True, help="configs/methods.yaml 中的预设名")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="覆盖配置，如 --set skills.verify=false")
    ap.add_argument("--llm", default=None, choices=["deepseek", "mock", "openai_compatible"], help="= llm.provider")
    ap.add_argument("--model", default=None, help="= llm.model")
    ap.add_argument("--n_train", type=int, default=2000, help="建库用的训练流长度（0 = 全部训练集）")
    ap.add_argument("--n_test", type=int, default=500, help="测试病例数（0 = 全部；固定随机顺序，所有方法用同一批）")
    ap.add_argument("--test_split", default="test", choices=["test", "val", "train"], help="train = 练习作答（检索时排除病例自己），供 learn_decision.py 学习")
    ap.add_argument("--order", default=None, choices=["random", "category"], help="= stream.order")
    ap.add_argument("--phase", default="all", choices=["build", "test", "all"])
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--load_memory", default=None, help="直接加载别的 run 的 memory_final.json（跳过建库）")
    ap.add_argument("--fresh", action="store_true", help="忽略已有断点，从头建库")
    args = ap.parse_args()

    overrides = list(args.set)
    if args.llm:
        overrides.append(f"llm.provider={args.llm}")
    if args.model:
        overrides.append(f"llm.model={args.model}")
    if args.order:
        overrides.append(f"stream.order={args.order}")
    cfg = load_config(args.config, method=args.method, overrides=overrides)
    if cfg.llm.provider == "mock":
        cfg.llm.cache_path = ""  # never cache mock responses
    order = cfg.stream.order
    model_tag = "mock" if cfg.llm.provider == "mock" else cfg.llm.model + ("-think" if cfg.llm.thinking else "")
    run_name = args.run_name or f"{args.method}_n{args.n_train or 'all'}_{order}_{model_tag}"
    run_dir = os.path.join(cfg.paths.runs_dir, run_name)
    if args.phase in ("build", "all") and not args.load_memory and not args.fresh:
        check_stream_resume(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(run_dir, "run.log"))
    save_json(os.path.join(run_dir, "config.json"), {**to_plain(cfg), "args": vars(args)})

    res = Resources(cfg)
    llm = build_llm(cfg)
    llm.call_log_path = os.path.join(run_dir, "llm_calls.jsonl")
    save_json(os.path.join(run_dir, "output_protocol.json"), {
        "version": OUTPUT_PROTOCOL_VERSION, "provider": cfg.llm.provider,
        "model": cfg.llm.model, "base_url": cfg.llm.base_url,
        "thinking": cfg.llm.thinking, "reasoning_effort": cfg.llm.reasoning_effort if cfg.llm.thinking else None,
        "temperature": None if cfg.llm.thinking else cfg.llm.temperature,
        "max_tokens": llm.output_budget(), "json_mode": cfg.llm.json_mode,
        "json_repair_retries": cfg.llm.get("json_repair_retries", 1),
        "accept_repaired_json": cfg.llm.get("accept_repaired_json", False),
        "prescribe_style": cfg.prescribe.style, "max_herbs": cfg.prescribe.max_herbs,
        "include_vocab": cfg.prescribe.include_vocab, "vocab_min_freq": cfg.prescribe.vocab_min_freq,
    })
    method = build_method(cfg, llm, res)
    train_ids = res.select("train", args.n_train or None, order)
    test_ids = res.select(args.test_split, args.n_test or None)
    LOG.info("方法 %s（%s）| 模型 %s | 训练流 %d 例（%s）| 测试 %d 例 | 输出目录 %s", args.method, cfg.method.description, model_tag, len(train_ids), order, len(test_ids), run_dir)

    try:
        # ------------------------------------------------------------- build
        if args.load_memory:
            load_method(method, args.load_memory)
            LOG.info("已加载记忆：%s → %s", args.load_memory, method.summary())
        elif args.phase in ("build", "all") and method.learns:
            method.warmup(train_ids)
            val_ids = res.select("val", int(cfg.stream.val_size)) if int(cfg.stream.val_every) > 0 else None
            run_stream(method, res, train_ids, run_dir, int(cfg.stream.window), val_ids, int(cfg.stream.val_every), resume=not args.fresh)
        elif args.phase == "test" and method.learns:
            for name in ("memory_final.json", "memory.json"):
                p = os.path.join(run_dir, name)
                if os.path.exists(p):
                    load_method(method, p)
                    LOG.info("测试使用 %s：%s", p, method.summary())
                    break
            else:
                LOG.warning("%s 中没有找到已建好的记忆，将用空记忆测试", run_dir)
            method.warmup(train_ids)

        # ------------------------------------------------------------- test
        if args.phase in ("test", "all"):
            agg = run_test(method, res, test_ids, run_dir, name=args.test_split)
            report = format_metrics(agg) + "\n\n" + format_output_stats(load_json(os.path.join(run_dir, f"{args.test_split}_output_stats.json")))
            print("\n" + report)
            with open(os.path.join(run_dir, f"{args.test_split}_metrics.md"), "w", encoding="utf-8") as f:
                f.write(report + "\n")
    finally:
        save_json(os.path.join(run_dir, "usage.json"), llm.usage.summary())
    print("\n" + llm.usage.pretty())
    if cfg.llm.provider == "mock":
        print("\n（mock 模式：费用是按真实提示长度与 DeepSeek 价格估算的；输出 token 按典型长度估计）")
    print(f"\n结果目录：{run_dir}")


if __name__ == "__main__":
    main()
