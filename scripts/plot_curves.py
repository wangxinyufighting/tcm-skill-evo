"""学习曲线：prequential F1（先开方后揭晓）随已学病例数的变化 + 记忆规模 + 累计费用。

python scripts/plot_curves.py --runs runs/ours_n2000_* runs/case_rag_n2000_* --out figures/curve.png
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import _bootstrap  # noqa: F401,E402


def mem_size(m: dict) -> float:
    if "n_active" in m:
        return m["n_active"]
    for k in ("memory_cases", "insights", "cheatsheet_chars"):
        if k in m:
            return m[k]
    return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="figures/learning_curve.png")
    ap.add_argument("--smooth", type=int, default=5, help="滑动平均窗口数")
    args = ap.parse_args()
    dirs = []
    for p in args.runs:
        dirs += sorted(glob.glob(p))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for d in dirs:
        path = os.path.join(d, "stream_log.jsonl")
        if not os.path.exists(path):
            continue
        logs = [json.loads(l) for l in open(path, encoding="utf-8")]
        logs = [l for l in logs if "f1" in l]
        if not logs:
            continue
        x = np.array([l["done"] for l in logs])
        y = np.array([l["f1"] for l in logs])
        k = max(1, args.smooth)
        ys = np.array([y[max(0, i - k + 1) : i + 1].mean() for i in range(len(y))])  # trailing mean keeps every point
        name = os.path.basename(d.rstrip("/"))
        axes[0].plot(x, ys, marker="o" if len(x) < 15 else None, ms=3, label=name)
        if any("val_f1" in l for l in logs):
            vx = [l["done"] for l in logs if "val_f1" in l]
            vy = [l["val_f1"] for l in logs if "val_f1" in l]
            axes[0].plot(vx, vy, "o--", ms=3, label=name + " (val)")
        axes[1].plot(x, [mem_size(l["memory"]) for l in logs], label=name)
        axes[2].plot(x, [l["usage"]["cost_usd"] for l in logs], label=name)
    axes[0].set_title(f"Prequential F1 (moving avg over {args.smooth} windows)")
    axes[0].set_xlabel("cases seen")
    axes[1].set_title("Memory size (active skills / cases / insights)")
    axes[1].set_xlabel("cases seen")
    axes[2].set_title("Cumulative API cost (USD)")
    axes[2].set_xlabel("cases seen")
    for a in axes:
        a.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"保存到 {args.out}")


if __name__ == "__main__":
    main()
