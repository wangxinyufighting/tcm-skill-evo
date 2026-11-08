"""Evaluation metrics for (variable-length, ordered) herb prescriptions."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Callable, Iterable

import numpy as np


def prf(pred: Iterable[str], gt: Iterable[str]) -> tuple[float, float, float]:
    P, G = set(pred), set(gt)
    if not P and not G:
        return 1.0, 1.0, 1.0
    tp = len(P & G)
    p = tp / len(P) if P else 0.0
    r = tp / len(G) if G else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def at_k(pred_ordered: list[str], gt: Iterable[str], k: int) -> tuple[float, float, float]:
    """Precision@k, Recall@k, NDCG@k using the model's own order (君→臣→佐→使)."""
    G = set(gt)
    top = pred_ordered[:k]
    hits = [1.0 if h in G else 0.0 for h in top]
    p = sum(hits) / k
    r = sum(hits) / len(G) if G else 0.0
    dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(G), k)))
    return p, r, (dcg / idcg if idcg else 0.0)


def soft_prf(pred: list[str], gt: list[str], same_cat: Callable[[str, str], bool], partial: float = 0.5) -> tuple[float, float, float]:
    """Exact matches count 1; remaining pred/gt pairs sharing an efficacy category count `partial` (one-to-one)."""
    P, G = list(dict.fromkeys(pred)), list(dict.fromkeys(gt))
    if not P or not G:
        return prf(P, G)
    exact = set(P) & set(G)
    rp = [h for h in P if h not in exact]
    rg = [h for h in G if h not in exact]
    used = set()
    soft = 0.0
    for a in rp:
        for j, b in enumerate(rg):
            if j not in used and same_cat(a, b):
                used.add(j)
                soft += partial
                break
    tp = len(exact) + soft
    p, r = tp / len(P), tp / len(G)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def bootstrap_ci(values: list[float], n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> dict:
    """Two-sided paired bootstrap test for mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert len(a) == len(b) and len(a) > 0
    d = a - b
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    diff = float(d.mean())
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"diff": diff, "ci95": (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))), "p_value": min(p, 1.0), "n": len(d)}


def evaluate_predictions(
    preds: dict[str, list[str]],
    cases: dict,  # cid -> Case
    herb_info=None,
    incompat=None,
    subsets: dict[str, set[str]] | None = None,
    ks: tuple[int, ...] = (5, 10, 20),
    extra: dict[str, dict] | None = None,
) -> dict:
    """Compute per-case metrics and aggregates (overall + subsets)."""
    per_case = {}
    for cid, pred in preds.items():
        gt = cases[cid].herbs
        p, r, f = prf(pred, gt)
        row = {"p": p, "r": r, "f1": f, "len_pred": len(set(pred)), "len_gt": len(set(gt))}
        for k in ks:
            pk, rk, nk = at_k(pred, gt, k)
            row[f"p@{k}"], row[f"r@{k}"], row[f"ndcg@{k}"] = pk, rk, nk
        if herb_info is not None:
            _, _, sf = soft_prf(pred, gt, herb_info.same_category)
            row["soft_f1"] = sf
        if incompat is not None:
            row["unsafe"] = 1.0 if incompat.violations(pred) else 0.0
            row["gt_unsafe"] = 1.0 if incompat.violations(gt) else 0.0
        if extra and cid in extra:
            row.update(extra[cid])
        row["len_abs_err"] = abs(row["len_pred"] - row["len_gt"])
        per_case[cid] = row

    def agg(ids: list[str]) -> dict:
        if not ids:
            return {"n": 0}
        keys = [k for k in per_case[ids[0]].keys() if isinstance(per_case[ids[0]][k], (int, float))]
        out = {"n": len(ids)}
        for k in keys:
            vals = [per_case[c][k] for c in ids if k in per_case[c]]
            out[k] = float(np.mean(vals)) if vals else 0.0
        f1s = [per_case[c]["f1"] for c in ids]
        out["f1_ci95"] = bootstrap_ci(f1s)
        return out

    all_ids = list(per_case)
    result = {"overall": agg(all_ids)}
    for name, ids in (subsets or {}).items():
        result[name] = agg([c for c in all_ids if c in ids])
    return {"aggregate": result, "per_case": per_case}


def default_subsets(cases: dict, cids: Iterable[str]) -> dict[str, set[str]]:
    cids = list(cids)
    subs: dict[str, set[str]] = defaultdict(set)
    for c in cids:
        fl = cases[c].flags or {}
        subs["dedup" if not fl.get("near_dup") else "near_dup"].add(c)
        b = fl.get("prev_bucket")
        if b:
            subs[f"prev_{b}"].add(c)
    return dict(subs)
