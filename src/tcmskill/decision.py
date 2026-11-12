"""Decision layer on top of an LLM prescription (no extra LLM calls).

* PrecedentIndex – exact-description precedents in the case memory. PTM copies many formulas
  between books with the same 主治 text. "unique" = every memory case with this exact
  description has the same herb set, so copying it is the strongest prediction available;
  "ambiguous" = the same description maps to different formulas (copying is weak there).
* KNNVoter       – similarity-weighted herb votes of the k nearest memory cases + a length target.
* fuse()         – re-rank / trim / complete the LLM herbs with the kNN votes.

All parameters are tuned on validation predictions (scripts/posthoc.py), never on test.
"""
from __future__ import annotations

import itertools
import re
from collections import Counter, defaultdict

import numpy as np

from .metrics import prf
from .textindex import TextIndex

_KEY_RE = re.compile(r"[\s，,。.、；;：:！!？?（）()《》“”\"'\-—…·]+")

PRECEDENT_KINDS = ("unique", "ambiguous", "none")


def text_key(text: str) -> str:
    return _KEY_RE.sub("", text or "")


class PrecedentIndex:
    def __init__(self, res, cids):
        self.res = res
        self.by_key: dict[str, list[str]] = defaultdict(list)
        for c in cids:
            k = text_key(res.text(res.cases[c]))
            if k:
                self.by_key[k].append(c)

    def lookup(self, case) -> tuple[str, list[str], list[str]]:
        """-> (kind, herbs_to_copy, precedent_cids); kind in {unique, ambiguous, none}."""
        grp = [c for c in self.by_key.get(text_key(self.res.text(case)), []) if c != case.cid]
        if not grp:
            return "none", [], []
        sets = {frozenset(self.res.cases[c].herbs) for c in grp}
        if len(sets) == 1:
            return "unique", list(self.res.cases[grp[0]].herbs), grp
        return "ambiguous", [], grp

    def kinds(self, cases) -> dict[str, str]:
        return {c.cid: self.lookup(c)[0] for c in cases}


class KNNVoter:
    def __init__(self, res, cids, k: int = 20):
        self.res, self.k = res, k
        self.index = TextIndex(res.vectorizer)
        self.index.add(list(cids), [res.text(res.cases[c]) for c in cids])

    def votes(self, cases) -> list[dict]:
        """Per case: {"votes": {herb: weight in [0,1]}, "L": weighted mean length, "sim1": top-1 similarity, "nb": [(cid, sim)]}."""
        nbrs = self.index.query([self.res.text(c) for c in cases], self.k, exclude=[{c.cid} for c in cases])
        out = []
        for nb in nbrs:
            if not nb:
                out.append({"votes": {}, "L": float(self.res.median_len), "sim1": 0.0, "nb": []})
                continue
            ws = sum(s for _, s in nb) or 1e-9
            v: Counter = Counter()
            for cid, s in nb:
                for h in self.res.cases[cid].herbs:
                    v[h] += s / ws
            L = sum(s * len(self.res.cases[cid].herbs) for cid, s in nb) / ws
            out.append({"votes": dict(v), "L": float(L), "sim1": float(nb[0][1]), "nb": [(cid, float(s)) for cid, s in nb]})
        return out


# ============================================================================ fusion
IDENTITY = {"gate": "off", "a": 1.0, "length": "llm", "delta": 0, "add_thr": None}


def fuse(llm_herbs: list[str], kv: dict, a: float = 1.0, length: str = "llm", delta: int = 0, add_thr: float | None = None) -> list[str]:
    """score(h) = a * llm(h) + (1 - a) * vote(h)
       llm(h)   = 1 - rank / (n + 1) for herbs the LLM wrote (its own 君→使 order), else 0
       vote(h)  = similarity-weighted share of the k nearest memory cases that use h
       add_thr  : herbs the LLM did not write may enter when vote(h) >= add_thr (None = never)
       length   : llm | knn | min | mean  (+ delta), at least 1 herb.
    With the default arguments the LLM prescription is returned unchanged."""
    if not llm_herbs and not kv.get("votes"):
        return []
    votes = kv.get("votes", {})
    n = len(llm_herbs)
    score: dict[str, float] = {}
    for i, h in enumerate(llm_herbs):
        score[h] = a * (1.0 - i / (n + 1)) + (1.0 - a) * votes.get(h, 0.0)
    if add_thr is not None:
        for h, v in votes.items():
            if h not in score and v >= add_thr:
                score[h] = (1.0 - a) * v
    if a >= 1.0 and length == "llm" and delta == 0 and add_thr is None:
        return list(llm_herbs)
    L_llm, L_knn = float(n or kv.get("L", 1.0)), float(kv.get("L", n or 1.0))
    L = {"llm": L_llm, "knn": L_knn, "min": min(L_llm, L_knn), "mean": 0.5 * (L_llm + L_knn)}[length]
    L = max(1, int(round(L + delta)))
    ranked = sorted(score, key=lambda h: (-score[h], llm_herbs.index(h) if h in llm_herbs else 999))
    return ranked[:L]


def decide(case, llm_herbs: list[str], kv: dict, prec: tuple, params: dict) -> tuple[list[str], str]:
    """Apply one parameter setting to one case. Returns (herbs, how)."""
    kind, copy, _ = prec
    if params.get("gate", "off") == "unique" and kind == "unique":
        return list(copy), "precedent"
    if not llm_herbs:  # failed / empty LLM answer: fall back to kNN
        return fuse([], kv, a=0.0, length="knn", add_thr=0.0), "knn_fallback"
    herbs = fuse(llm_herbs, kv, params.get("a", 1.0), params.get("length", "llm"), int(params.get("delta", 0)), params.get("add_thr"))
    return herbs, ("llm" if herbs == list(llm_herbs) else "fused")


def param_grid() -> list[dict]:
    grid = []
    for gate in ("off", "unique"):
        grid.append({**IDENTITY, "gate": gate})
        for a, length, delta, add_thr in itertools.product((0.8, 0.6, 0.4, 0.2), ("llm", "knn", "min", "mean"), (-2, -1, 0, 1), (None, 0.7, 0.5)):
            grid.append({"gate": gate, "a": a, "length": length, "delta": delta, "add_thr": add_thr})
    return grid


def tune(items: list[dict], grid: list[dict] | None = None) -> tuple[dict, float, list[tuple[float, dict]]]:
    """items: {"case", "llm", "kv", "prec"}; maximises mean F1. Ties -> simpler setting (earlier in grid)."""
    grid = grid or param_grid()
    scored = []
    for p in grid:
        f = float(np.mean([prf(decide(it["case"], it["llm"], it["kv"], it["prec"], p)[0], it["case"].herbs)[2] for it in items])) if items else 0.0
        scored.append((f, p))
    best_f, best_p = max(scored, key=lambda x: x[0])  # max keeps the first maximum
    return best_p, best_f, sorted(scored, key=lambda x: -x[0])[:10]


def describe(p: dict) -> str:
    if p.get("gate") == "off" and p.get("a", 1.0) >= 1.0 and p.get("length") == "llm" and not p.get("delta") and p.get("add_thr") is None:
        return "不改动（原 LLM 处方）"
    parts = []
    if p.get("gate") == "unique":
        parts.append("唯一先例直接照抄")
    if not (p.get("a", 1.0) >= 1.0 and p.get("length") == "llm" and not p.get("delta") and p.get("add_thr") is None):
        L = {"llm": "LLM 味数", "knn": "kNN 味数", "min": "两者较小", "mean": "两者平均"}[p.get("length", "llm")]
        parts.append(f"融合 a={p.get('a')}（LLM 权重）、味数={L}{int(p.get('delta', 0)):+d}、补药阈值={p.get('add_thr')}")
    return "；".join(parts)
