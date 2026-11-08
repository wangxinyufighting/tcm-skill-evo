"""Step 1：数据准备 + 数据报告。

python scripts/prepare_data.py                       # 默认 gt_mode=fmm_alias
python scripts/prepare_data.py --set data.gt_mode=ptm --set paths.processed_dir=data/processed_ptm
"""
import argparse

import _bootstrap  # noqa: F401

from tcmskill.config import load_config
from tcmskill.data import prepare
from tcmskill.utils import setup_logging


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. data.gt_mode=fmm")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config, overrides=args.set)
    stats = prepare(cfg, cfg.paths.ptm_dir, cfg.paths.processed_dir, cfg.paths.kb_dir)
    print("\n".join(f"{k}: {v}" for k, v in stats.items()))
    print(f"\n报告：{cfg.paths.processed_dir}/report.md")


if __name__ == "__main__":
    main()
