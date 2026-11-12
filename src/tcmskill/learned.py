"""Learned decision layer: "decision skills" learned from feedback.

Candidates per case = herbs of every LLM run ∪ herbs voted by the k nearest memory cases ∪ unique-precedent herbs
(∪ herbs suggested by the skill library). Every (case, herb) pair gets features from these groups:

  retr    retrieval + precedent, no LLM: kNN vote, share among the top-5 neighbours, in the top-1 neighbour, best
          similarity of a top-5 neighbour using it, log prior frequency, top-1 similarity, kNN length, precedent flags
  llm{j}  one group per LLM run: written by the run, its rank in the run, the run's prescription length
  skill   skill library: strongest firing 加减 rule for the herb (log lift, confidence), share of the top-5
          neighbours whose 方族 core contains it

A gradient-boosting (or logistic) model predicts P(herb in the original formula); each case gets the top-k herbs that
maximise the plug-in expected F1  2·Σ_{i≤k} p_i / (k + Σ_all p).  Trained by cross-fitting on validation predictions,
or on practice predictions over training cases (run_method.py --test_split train): the agent practises, sees the
original formula, and learns which of its own (and its memory's) proposals to keep.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .decision import KNNVoter, PrecedentIndex

RETR = ["knn_vote", "top5_share", "in_top1", "max_sim_top5", "log_prior", "sim1", "knn_len", "prec_herb", "prec_unique", "prec_ambiguous"]
LLM_F = ["in", "rank", "len"]
SKILL = ["rule_loglift", "rule_conf", "fam_core_share"]


@dataclass
class FeatureTable:
    X: np.ndarray
    y: np.ndarray
    case_idx: np.ndarray
    herbs: list
    names: list
    src: dict
    cases: list
    kinds: list  # precedent kind per case

    def columns(self, groups) -> list[int]:
        keep = []
        for i, n in enumerate(self.names):
            g = "retr" if n in RETR else ("skill" if n in SKILL else n.split("_")[0])
            if g in groups:
                keep.append(i)
        return keep

    def rows(self, groups) -> np.ndarray:
        m = self.src["knn"] | self.src["prec"]
        for g in groups:
            if g != "retr" and g in self.src:
                m = m | self.src[g]
        return m


class FeatureBuilder:
    def __init__(self, res, memory_cids, k: int = 20, skill_memory: dict | None = None, cfg=None):
        self.res = res
        self.mem = set(memory_cids)
        self.voter = KNNVoter(res, memory_cids, k)
        self.pidx = PrecedentIndex(res, memory_cids)
        # Global herb counts over the whole memory. Deliberately NOT leave-one-out for practice cases: subtracting a
        # training case's own herbs moves the feature only for its original-formula herbs, i.e. it encodes the label.
        self.prior = Counter(h for c in memory_cids for h in res.cases[c].herbs)
        self.skills = None
        if skill_memory is not None:
            from .skills import SkillLibrary, SkillMatcher
            from .stats import SeenStore

            lib = SkillLibrary()
            lib.load_state_dict(skill_memory["library"])
            store = SeenStore(res)
            store.load_state_dict(skill_memory.get("store", {"cids": list(memory_cids)}))
            links = defaultdict(list)
            for s in lib.active("formula"):
                for cid in s.evidence:
                    links[cid].append(s)
            self.skills = (lib.active("modification"), SkillMatcher(store, cfg.skills.match), links)

    def build(self, cases: list, llm_runs: list[dict]) -> FeatureTable:
        """llm_runs: one {cid: [herbs in the run's own order]} per LLM run (missing or failed = empty)."""
        kvs = self.voter.votes(cases)
        n_llm = len(llm_runs)
        names = RETR + [f"llm{j}_{f}" for j in range(n_llm) for f in LLM_F] + (SKILL if self.skills else [])
        rows, ys, ci, hs, kinds = [], [], [], [], []
        src = defaultdict(list)
        for i, (c, kv) in enumerate(zip(cases, kvs)):
            kind, prec_h, _ = self.pidx.lookup(c)
            kinds.append(kind)
            prec_set = set(prec_h) if kind == "unique" else set()
            nb5 = kv["nb"][:5]
            top1 = set(self.res.cases[nb5[0][0]].herbs) if nb5 else set()
            cnt5, maxsim = Counter(), {}
            for cid, s in nb5:
                for h in self.res.cases[cid].herbs:
                    cnt5[h] += 1
                    maxsim[h] = max(maxsim.get(h, 0.0), s)
            gt = set(c.herbs)
            llm = [list(r.get(c.cid) or []) for r in llm_runs]
            rule: dict = {}
            fam: Counter = Counter()
            if self.skills:
                rules, matcher, links = self.skills
                text, sym = self.res.text(c), set(c.symptoms)
                for r in rules:
                    if matcher.match(r, text, sym) >= matcher.fire:
                        ll, cf = math.log(max(float(r.vstats.get("lift", 1.0)), 1.0)), float(r.vstats.get("conf", 0.0))
                        for h in r.add:
                            a, b = rule.get(h, (0.0, 0.0))
                            rule[h] = (max(a, ll), max(b, cf))
                for cid, _ in nb5:
                    for s in links.get(cid, []):
                        for h in s.core_herbs():
                            fam[h] += 1
            cand = set(kv["votes"]) | prec_set | {h for lst in llm for h in lst} | set(rule) | set(fam)
            for h in sorted(cand):
                f = [kv["votes"].get(h, 0.0), cnt5[h] / 5.0, float(h in top1), maxsim.get(h, 0.0),
                     math.log1p(self.prior[h]), kv["sim1"], kv["L"],
                     float(h in prec_set), float(kind == "unique"), float(kind == "ambiguous")]
                for lst in llm:
                    n = len(lst)
                    f += [float(h in lst), (1.0 - lst.index(h) / (n + 1)) if h in lst else 0.0, float(n)]
                if self.skills:
                    a, b = rule.get(h, (0.0, 0.0))
                    f += [a, b, fam[h] / 5.0]
                rows.append(f)
                ys.append(float(h in gt))
                ci.append(i)
                hs.append(h)
                src["knn"].append(h in kv["votes"])
                src["prec"].append(h in prec_set)
                for j, lst in enumerate(llm):
                    src[f"llm{j}"].append(h in lst)
                src["skill"].append(h in rule or h in fam)
        X = np.asarray(rows, dtype=np.float64).reshape(-1, len(names))
        return FeatureTable(X, np.asarray(ys), np.asarray(ci, dtype=np.int64), hs, names,
                            {k: np.asarray(v, dtype=bool) for k, v in src.items()}, cases, kinds)


# ============================================================================ model + selection
def make_model(kind: str = "gb"):
    if kind == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0)


def select_topk(p: np.ndarray, herbs: list, kmax: int = 30) -> list:
    if len(p) == 0:
        return []
    order = np.argsort(-p, kind="stable")
    tot = float(p.sum())
    best, k, run = -1.0, 1, 0.0
    for j in range(min(len(order), kmax)):
        run += float(p[order[j]])
        v = 2.0 * run / (j + 1 + tot)
        if v > best:
            best, k = v, j + 1
    return [herbs[i] for i in order[:k]]


def _predict(model, T: FeatureTable, cols, mask) -> dict[int, list]:
    idx = np.flatnonzero(mask)
    out: dict[int, list] = {}
    if len(idx) == 0:
        return out
    p = model.predict_proba(T.X[idx][:, cols])[:, 1]
    by = defaultdict(list)
    for r, pr in zip(idx, p):
        by[int(T.case_idx[r])].append((pr, T.herbs[r]))
    for c, lst in by.items():
        out[c] = select_topk(np.array([a for a, _ in lst]), [b for _, b in lst])
    return out


def fit(T: FeatureTable, groups, kind: str = "gb", case_subset=None):
    cols, mask = T.columns(groups), T.rows(groups)
    if case_subset is not None:
        mask = mask & np.isin(T.case_idx, np.asarray(list(case_subset)))
    return make_model(kind).fit(T.X[mask][:, cols], T.y[mask])


def apply(model, T: FeatureTable, groups) -> list[list]:
    pred = _predict(model, T, T.columns(groups), T.rows(groups))
    return [pred.get(i, []) for i in range(len(T.cases))]


def crossfit(T: FeatureTable, groups, kind: str = "gb", folds: int = 5, seed: int = 0) -> list[list]:
    n = len(T.cases)
    fold = np.random.default_rng(seed).permutation(n) % folds
    cols, mask = T.columns(groups), T.rows(groups)
    out = [[] for _ in range(n)]
    for f in range(folds):
        tr = mask & (fold[T.case_idx] != f)
        te = mask & (fold[T.case_idx] == f)
        m = make_model(kind).fit(T.X[tr][:, cols], T.y[tr])
        for c, h in _predict(m, T, cols, te).items():
            out[c] = h
    return out


def lr_weights(T: FeatureTable, groups) -> list[tuple[str, float]]:
    """Standardised logistic-regression weights: a readable summary of what the decision layer learned."""
    m = fit(T, groups, kind="lr")
    coef = m[-1].coef_[0]
    names = [T.names[i] for i in T.columns(groups)]
    return sorted(zip(names, coef.tolist()), key=lambda x: -abs(x[1]))
