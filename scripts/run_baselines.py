"""非 LLM 基线：freq / knn（描述）/ knn_kw（关键词）/ mlp。

python scripts/run_baselines.py --methods freq,knn,knn_kw --n_test 500            # 记忆 = 全部训练集
python scripts/run_baselines.py --methods knn --n_train 2000 --n_test 500         # 记忆 = 与 LLM 方法相同的 2000 例训练流
python scripts/run_baselines.py --methods mlp --n_test 500                         # 较慢（几分钟）
"""
import argparse
import os

import _bootstrap  # noqa: F401

from tcmskill.baselines import FreqBaseline, KNNBaseline, MLPBaseline
from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.runner import evaluate_run, format_metrics
from tcmskill.utils import save_json, setup_logging, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="freq,knn,knn_kw")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--n_train", type=int, default=0, help="0 = 全部训练集；>0 = 与 LLM 方法相同的训练流前 N 例")
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--test_split", default="test", choices=["test", "val"])
    ap.add_argument("--k", type=int, default=None)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, overrides=args.set)
    res = Resources(cfg)
    train_ids = res.select("train", args.n_train or None, cfg.stream.order)
    test_ids = res.select(args.test_split, args.n_test or None)
    cases = res.get(test_ids)
    for m in args.methods.split(","):
        m = m.strip()
        if m == "freq":
            model = FreqBaseline(res).fit(train_ids)
        elif m == "knn":
            model = KNNBaseline(res, k=args.k or int(cfg.knn.k), feature="description").fit(train_ids)
        elif m == "knn_kw":
            model = KNNBaseline(res, k=args.k or int(cfg.knn.k), feature="keywords").fit(train_ids)
        elif m == "mlp":
            model = MLPBaseline(res).fit(train_ids, res.select("val", 500))
        else:
            raise SystemExit(f"unknown baseline {m}")
        run_dir = os.path.join(cfg.paths.runs_dir, f"{m}_n{args.n_train or 'all'}")
        os.makedirs(run_dir, exist_ok=True)
        preds = model.predict(cases)
        write_jsonl(os.path.join(run_dir, f"{args.test_split}_preds.jsonl"), [p.to_dict() for p in preds])
        agg = evaluate_run(res, preds, run_dir, args.test_split)
        save_json(os.path.join(run_dir, "config.json"), {"method": m, "n_train": len(train_ids), "n_test": len(test_ids)})
        print(f"\n=== {m}（训练 {len(train_ids)} 例）===\n" + format_metrics(agg))


if __name__ == "__main__":
    main()
