"""SeenStore: everything the agent has observed so far (cases with revealed GT).

Used for (a) IDF weights of trigger terms, (b) statistical verification of candidate skills
("LLM proposes, data verifies"), (c) historical neighbours for multi-case induction,
(d) case retrieval for case-based baselines / hybrid mode.
Only cases whose GT has been revealed are added (no leakage).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Iterable

from .textindex import TextIndex


class SeenStore:
    def __init__(self, res):
        self.res = res
        self.cids: list[str] = []
        self.texts: list[str] = []
        self.symsets: list[frozenset] = []
        self.herbsets: list[frozenset] = []
        self.cats: list[int] = []
        self.pos: dict[str, int] = {}
        self.herb_post: dict[str, set[int]] = defaultdict(set)
        self._term_post: dict[str, set[int]] = {}
        self.index = TextIndex(res.vectorizer)

    # ------------------------------------------------------------------ building
    @property
    def n(self) -> int:
        return len(self.cids)

    def add(self, cases: Iterable) -> None:
        new_idx, ids, txts = [], [], []
        for c in cases:
            if c.cid in self.pos:
                continue
            i = len(self.cids)
            t = self.res.text(c)
            self.pos[c.cid] = i
            self.cids.append(c.cid)
            self.texts.append(t)
            self.symsets.append(frozenset(c.symptoms))
            self.herbsets.append(frozenset(c.herbs))
            self.cats.append(c.cat)
            for h in self.herbsets[-1]:
                self.herb_post[h].add(i)
            new_idx.append(i)
            ids.append(c.cid)
            txts.append(t)
        if not new_idx:
            return
        for term, post in self._term_post.items():
            for i in new_idx:
                if self.present_idx(term, i):
                    post.add(i)
        self.index.add(ids, txts)

    # ------------------------------------------------------------------ terms
    @staticmethod
    def present(term: str, text: str, symset) -> bool:
        return term in text or term in symset

    def present_idx(self, term: str, i: int) -> bool:
        return term in self.texts[i] or term in self.symsets[i]

    def term_post(self, term: str) -> set[int]:
        post = self._term_post.get(term)
        if post is None:
            post = {i for i in range(self.n) if self.present_idx(term, i)}
            self._term_post[term] = post
        return post

    def df(self, term: str) -> int:
        return len(self.term_post(term))

    def idf(self, term: str) -> float:
        return math.log((self.n + 1) / (self.df(term) + 1)) + 1.0

    def herb_p(self, h: str) -> float:
        return len(self.herb_post.get(h, ())) / max(self.n, 1)

    def cases_with_all(self, terms: list[str]) -> set[int]:
        if not terms:
            return set()
        posts = sorted((self.term_post(t) for t in terms), key=len)
        out = set(posts[0])
        for p in posts[1:]:
            out &= p
            if not out:
                break
        return out

    def rule_stats(self, terms: list[str], herb: str, within: set[int] | None = None) -> dict:
        """Statistics of the rule (all `terms` present) => `herb` in GT."""
        X = self.cases_with_all(terms)
        hp = self.herb_post.get(herb, set())
        if within is not None:
            X = X & within
            base_n = max(len(within), 1)
            p_y = len(hp & within) / base_n
        else:
            p_y = len(hp) / max(self.n, 1)
        n_x = len(X)
        n_xy = len(X & hp)
        conf = n_xy / n_x if n_x else 0.0
        lift = conf / p_y if p_y > 0 else 0.0
        return {"n_x": n_x, "n_xy": n_xy, "conf": round(conf, 4), "p_y": round(p_y, 4), "lift": round(lift, 3)}

    # ------------------------------------------------------------------ neighbours
    def neighbors_by_herbs(self, herbs: Iterable[str], k: int, min_jacc: float, exclude: set[str] = frozenset()) -> list[tuple[int, float]]:
        hs = set(herbs)
        if not hs or k <= 0:
            return []
        cnt: Counter = Counter()
        for h in hs:
            for i in self.herb_post.get(h, ()):
                cnt[i] += 1
        out = []
        for i, inter in cnt.items():
            if self.cids[i] in exclude:
                continue
            j = inter / (len(hs) + len(self.herbsets[i]) - inter)
            if j >= min_jacc:
                out.append((i, j))
        out.sort(key=lambda x: -x[1])
        return out[:k]

    def neighbors_by_text(self, texts: list[str], k: int, exclude: list[set[str]] | None = None):
        return self.index.query(texts, k, exclude)

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"cids": list(self.cids)}

    def load_state_dict(self, d: dict) -> None:
        self.add(self.res.get(d.get("cids", [])))
