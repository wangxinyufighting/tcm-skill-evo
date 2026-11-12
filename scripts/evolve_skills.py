"""开方技能进化：让 LLM 在练习中自己总结开方规律（组内对比 + 验证门控），并评测技能的作用。

  # 1) 练习：训练流前 480 例，每步 16 例；训练流最后 120 例做门控（都不碰 val / test）
  python scripts/evolve_skills.py train --run_name sk_v1

  # 2) 同一提示词、同一模型，有 / 无技能在 val 上对比（论文主表）
  python scripts/evolve_skills.py eval --book runs/sk_v1/skillbook_final.json --split val --n 1000 --out runs/s5_skill_val1000
  python scripts/evolve_skills.py eval --empty --split val --n 1000 --out runs/s5_noskill_val1000

  # 3) 进化曲线：每个被接受的技能版本在 val 前 300 例上的 F1
  python scripts/evolve_skills.py curve --run_name sk_v1 --split val --n 300

智能体（作答 / 门控 / 评测）一律不开思考；只有优化器默认开思考（skillbook.optimizer_thinking）。
中途断了，重跑同一条 train 命令会从上次的步数接着跑（已有的请求走缓存）。
并发数：configs/default.yaml 里的 llm.workers，或每条命令加 --set llm.workers=32。
请求在客户端重试耗尽后仍失败的病例不会按 F1=0 计分：先整体重跑（skillbook.retry_passes 轮），仍失败则本步不提交并退出。
"""
import argparse
import copy
import glob
import os
import re
import shutil

import numpy as np

import _bootstrap  # noqa: F401

from tcmskill.config import load_config, to_plain
from tcmskill.data import Resources
from tcmskill.decision import KNNVoter, PrecedentIndex
from tcmskill.llm import FatalLLMError, LLMRequestError, build_llm
from tcmskill.metrics import prf
from tcmskill.prescriber import Prediction
from tcmskill.runner import evaluate_run, format_metrics
from tcmskill.skillbook import Evolver, SkillBook, evaluate_book
from tcmskill.utils import load_json, save_json, setup_logging, write_jsonl


def make_llm(cfg, run_dir, thinking: bool, temperature=None, tag=""):
    c = copy.deepcopy(cfg)
    c.llm.thinking = bool(thinking)
    if temperature is not None:
        c.llm.temperature = float(temperature)
    if c.llm.provider == "mock":
        c.llm.cache_path = ""
    llm = build_llm(c)
    llm.call_log_path = os.path.join(run_dir, f"llm_calls{tag}.jsonl")
    return llm


def usage_of(llms: dict) -> dict:
    out = {k: v.usage.summary() for k, v in llms.items() if v is not None}
    out["total"] = {"cost_usd": round(sum(float(u.get("total", {}).get("cost_usd", 0.0)) for u in out.values()), 4)}
    return out


def to_preds(res, outs):
    return [Prediction(cid=o["cid"], herbs=o["herbs"], ok=o["ok"], error=o.get("error"), used_skills=o.get("used", []), oov=o.get("oov", []),
                       json_status=o.get("json_status", "unknown"), finish_reason=o.get("finish_reason"), request_failed=o.get("request_failed", False),
                       unsafe=res.incompat.violations(o["herbs"]),
                       entries=[{"name": h, "role": "", "source": "skill", "targets": []} for h in o["herbs"]]) for o in outs]


def subset_f1(res, cases, outs):
    pidx = PrecedentIndex(res, res.select("train"))
    f = np.array([prf(o["herbs"], c.herbs) for o, c in zip(outs, cases)])
    kinds = [pidx.lookup(c)[0] for c in cases]
    pick = lambda m: float(f[np.array(m, bool), 2].mean()) if any(m) else float("nan")
    return {"f1": float(f[:, 2].mean()), "p": float(f[:, 0].mean()), "r": float(f[:, 1].mean()),
            "unique": pick([k == "unique" for k in kinds]), "none": pick([k == "none" for k in kinds]),
            "rare": pick([(c.flags or {}).get("prev_bucket") == "rare" for c in cases]),
            "len": float(np.mean([len(o["herbs"]) for o in outs]))}


def cmd_train(args, cfg):
    run_dir = os.path.join(cfg.paths.runs_dir, args.run_name)
    if args.fresh and os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    setup_logging(log_file=os.path.join(run_dir, "run.log"))
    save_json(os.path.join(run_dir, "config.json"), {**to_plain(cfg), "args": vars(args)})
    res = Resources(cfg)
    sc = cfg.skillbook
    llms = {"agent": make_llm(cfg, run_dir, False),
            "explore": make_llm(cfg, run_dir, False, sc.explore_temperature, "_explore") if float(sc.explore_temperature) > 0 else None,
            "optimizer": make_llm(cfg, run_dir, bool(sc.optimizer_thinking), None, "_opt")}
    ev = Evolver(res, cfg, run_dir, llms["agent"], llms["explore"], llms["optimizer"])
    resumed = ev.load()
    steps = int(np.ceil(len(ev.practice) / int(sc.batch))) * int(sc.epochs)
    print(f"== 练习 {len(ev.practice)} 例 × {sc.epochs} 轮，每步 {sc.batch} 例，共 {steps} 步；门控留出 {len(ev.sel)} 例"
          + (f"；从第 {ev.step} 步继续，当前 {len(ev.book.rules)} 条技能" if resumed else ""))

    names = {"weak_support": "支持不足", "counter_examples": "反例多", "not_confirmed": "复核未过", "screened_out": "初筛", "rejected": "", "too_few_affected": "影响太少", "duplicate": "重复",
             "unknown_id": "编号不存在", "book_full": "已满", "empty_text": "空"}

    def show(row):
        g = row.get("gate", {})
        if g:   # joint gate / no gate (ablations)
            gate = f"Δ{g.get('delta', 0):+.3f} t={g.get('t', 0)} n={g.get('n_affected', 0)}"
        else:   # per-edit gate: ✓ / ✗ for every proposed edit
            marks = []
            for tr in row.get("trials", []):
                tg, o = tr.get("gate") or {}, tr["outcome"]
                cf = tg.get("confirm") or {}
                again = f"/复核{cf['delta']:+.3f}" if cf.get("tested") else ""
                if o == "accepted":
                    marks.append(f"✓Δ{tg['delta']:+.3f}(t={tg['t']},n={tg['n_tested']}){again}")
                elif tg:
                    marks.append(f"✗Δ{tg.get('delta', 0):+.3f}{again}{names.get(o, o)}")
                else:
                    marks.append(f"✗{names.get(o, o)}")
            gate = " ".join(marks)
        retried = f"｜重跑失败请求 {row['retried']} 个" if row.get("retried") else ""
        print(f"[步 {row['step']:3d}] 智能体 {row['f1_agent']:.3f}｜照抄C1 {row['f1_copy_c1']:.3f}｜投票 {row['f1_knn']:.3f}｜"
              f"{row.get('outcome', '')} {gate}｜技能 {row.get('n_rules_after', row['n_rules'])} 条｜${row['cost_usd']:.2f}{retried}", flush=True)

    ev.train(int(sc.epochs), seed=int(cfg.get("seed", 0)), max_steps=args.max_steps, on_step=show)
    save_json(os.path.join(run_dir, "skillbook_final.json"), ev.book.state_dict())
    with open(os.path.join(run_dir, "skillbook_final.md"), "w", encoding="utf-8") as f:
        f.write(f"# 进化得到的开方技能（{args.run_name}，{len(ev.versions)} 个版本，第 {ev.step} 步）\n\n" + ev.book.to_markdown())
    save_json(os.path.join(run_dir, "usage.json"), usage_of(llms))
    n_acc = sum(len(v.get("edits", [])) for v in ev.versions)
    print(f"\n接受 {n_acc} 条编辑（{len(ev.versions)} 个版本），门控拒绝 {sum(len(r['edits']) for r in ev.rejected)} 条；最终 {len(ev.book.rules)} 条技能 → {run_dir}/skillbook_final.md")
    print(f"费用 ${usage_of(llms)['total']['cost_usd']:.2f}")


def cmd_eval(args, cfg):
    os.makedirs(args.out, exist_ok=True)
    setup_logging(log_file=os.path.join(args.out, "run.log"))
    res = Resources(cfg)
    book = SkillBook() if args.empty else SkillBook.from_state(load_json(args.book))
    llm0 = make_llm(cfg, args.out, False)
    cases = res.get(res.select(args.split, args.n or None))
    outs = evaluate_book(res, cfg, book, cases, llm0)
    preds = to_preds(res, outs)
    write_jsonl(os.path.join(args.out, f"{args.split}_preds.jsonl"), [p.to_dict() for p in preds])
    agg = evaluate_run(res, preds, args.out, args.split)
    save_json(os.path.join(args.out, "skillbook.json"), book.state_dict())
    save_json(os.path.join(args.out, "config.json"), {**to_plain(cfg), "args": vars(args), "book_signature": book.signature(), "n_rules": len(book.rules)})
    save_json(os.path.join(args.out, "usage.json"), usage_of({"agent": llm0}))
    print(format_metrics(agg))
    s = subset_f1(res, cases, outs)
    print(f"\n{'无技能' if args.empty else f'{len(book.rules)} 条技能'}：F1 {s['f1']:.3f}｜唯一先例 {s['unique']:.3f}｜无先例 {s['none']:.3f}｜rare {s['rare']:.3f}｜{s['len']:.1f} 味"
          f"｜引用技能的病例 {np.mean([bool(o['used']) for o in outs]):.1%}｜费用 ${usage_of({'a': llm0})['total']['cost_usd']:.2f}\n写入 {args.out}")


def cmd_curve(args, cfg):
    run_dir = os.path.join(cfg.paths.runs_dir, args.run_name)
    setup_logging(log_file=os.path.join(run_dir, "curve.log"))
    res = Resources(cfg)
    state = load_json(os.path.join(run_dir, "state.json"))
    files = sorted(glob.glob(os.path.join(run_dir, "versions", "book_v*.json")), key=lambda p: int(re.findall(r"v(\d+)", p)[-1]))
    meta = {v["v"]: v for v in state.get("versions", [])}
    books = [(0, 0, SkillBook())] + [(int(re.findall(r"v(\d+)", p)[-1]), meta.get(int(re.findall(r"v(\d+)", p)[-1]), {}).get("step", "?"),
                                      SkillBook.from_state(load_json(p))) for p in files]
    if args.every > 1 and len(books) > 2:
        books = [b for i, b in enumerate(books) if i % args.every == 0 or i == len(books) - 1]
    llm0 = make_llm(cfg, run_dir, False, tag="_curve")
    cases = res.get(res.select(args.split, args.n or None))
    voter = KNNVoter(res, res.select("train"), k=20)
    rows, base = [], None
    for v, step, book in books:
        outs = evaluate_book(res, cfg, book, cases, llm0, voter)
        s = subset_f1(res, cases, outs)
        base = s["f1"] if base is None else base
        rows.append(f"| v{v} | {step} | {len(book.rules)} | {s['f1']:.3f} | {s['f1'] - base:+.3f} | {s['unique']:.3f} | {s['none']:.3f} | {s['rare']:.3f} | {s['p']:.3f} | {s['r']:.3f} | {s['len']:.1f} |")
        print(rows[-1], flush=True)
    head = (f"# 技能进化曲线：{args.run_name}（{args.split} 前 {len(cases)} 例；v0 = 无技能，同一提示词）\n\n"
            "| 版本 | 接受于第几步 | 技能数 | F1 | 相对 v0 | 唯一先例 | 无先例 | rare | P | R | 平均味数 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    with open(os.path.join(run_dir, f"curve_{args.split}{len(cases)}.md"), "w", encoding="utf-8") as f:
        f.write(head + "\n" + "\n".join(rows) + f"\n\n费用 ${usage_of({'a': llm0})['total']['cost_usd']:.2f}\n")
    print(f"\n写入 {run_dir}/curve_{args.split}{len(cases)}.md")


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    ap.add_argument("cmd", choices=["train", "eval", "curve"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--llm", default=None, choices=["deepseek", "mock", "openai_compatible"])
    ap.add_argument("--run_name", default="sk_v1")
    ap.add_argument("--fresh", action="store_true", help="train：删掉同名 run 重新开始")
    ap.add_argument("--max_steps", type=int, default=None, help="train：最多跑到第几步（便于先小规模试跑）")
    ap.add_argument("--book", default=None, help="eval：技能文件（skillbook_final.json 或 versions/book_vN.json）")
    ap.add_argument("--empty", action="store_true", help="eval：不带任何技能（同一提示词的对照）")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--every", type=int, default=1, help="curve：每隔几个版本评测一次")
    args = ap.parse_args()
    overrides = list(args.set) + ([f"llm.provider={args.llm}"] if args.llm else [])
    cfg = load_config(args.config, overrides=overrides)
    if args.cmd == "eval" and (not args.out or (not args.book and not args.empty)):
        raise SystemExit("eval 需要 --out，以及 --book 或 --empty")
    try:
        {"train": cmd_train, "eval": cmd_eval, "curve": cmd_curve}[args.cmd](args, cfg)
    except FatalLLMError as e:
        raise SystemExit(f"\n{e}") from None
    except LLMRequestError as e:
        raise SystemExit(f"\n请求多次重试后仍失败：{e}\n没有写入不完整的结果。网络恢复（或调低 llm.workers）后重新运行同一条命令即可续跑；"
                         "已成功的请求走缓存，不重复收费。") from None


if __name__ == "__main__":
    main()
