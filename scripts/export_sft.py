"""导出微调基线数据（DeepSeek API 不能微调，微调基线需要本地开源模型，如 Qwen3-8B + LLaMA-Factory）。

python scripts/export_sft.py --out data/sft                         # train/val/test（OpenAI messages 格式）
python scripts/export_sft.py --out data/sft --n_train 2000          # 只用与 LLM 方法相同的 2000 例训练流
python scripts/export_sft.py --out data/sft_cat --order category --chunks 5   # 按类别顺序切块，做持续微调/遗忘实验

微调模型在 test.jsonl 上的输出整理成 {"cid":..., "text":"药1、药2、..."} 每行一条，然后：
python scripts/evaluate.py --pred outputs.jsonl --out runs/sft_xxx
"""
import argparse
import os

import _bootstrap  # noqa: F401

from tcmskill import prompts as P
from tcmskill.config import load_config
from tcmskill.data import Resources
from tcmskill.utils import write_jsonl

SFT_SYSTEM = "你是一名经验丰富的中医临床医师。根据病例的主治/症状描述开具中药处方，只输出药名，用顿号分隔，按重要性排列。"


def row(res, cid, fmt):
    c = res.cases[cid]
    user = P.prescribe_user(res.text(c)).replace("请先辨证，再给出处方 JSON。", "请开方。")
    ans = "、".join(c.herbs)
    if fmt == "alpaca":
        return {"cid": cid, "instruction": SFT_SYSTEM, "input": user, "output": ans}
    return {"cid": cid, "messages": [{"role": "system", "content": SFT_SYSTEM}, {"role": "user", "content": user}, {"role": "assistant", "content": ans}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sft")
    ap.add_argument("--n_train", type=int, default=0)
    ap.add_argument("--n_test", type=int, default=0)
    ap.add_argument("--order", default="random", choices=["random", "category"])
    ap.add_argument("--chunks", type=int, default=0)
    ap.add_argument("--format", default="messages", choices=["messages", "alpaca"])
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    res = Resources(cfg)
    tr = res.select("train", args.n_train or None, args.order)
    os.makedirs(args.out, exist_ok=True)
    write_jsonl(os.path.join(args.out, "train.jsonl"), (row(res, c, args.format) for c in tr))
    write_jsonl(os.path.join(args.out, "val.jsonl"), (row(res, c, args.format) for c in res.select("val")))
    write_jsonl(os.path.join(args.out, "test.jsonl"), (row(res, c, args.format) for c in res.select("test", args.n_test or None)))
    if args.chunks:
        size = (len(tr) + args.chunks - 1) // args.chunks
        for i in range(args.chunks):
            part = tr[i * size : (i + 1) * size]
            write_jsonl(os.path.join(args.out, "stream_chunks", f"chunk_{i}.jsonl"), (row(res, c, args.format) for c in part))
    print(f"写入 {args.out}：train {len(tr)} / val {len(res.select('val'))} / test {len(res.select('test', args.n_test or None))}")


if __name__ == "__main__":
    main()
