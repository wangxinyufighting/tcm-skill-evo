"""Incremental TF-IDF (char n-gram) cosine index."""
from __future__ import annotations

import numpy as np
from scipy import sparse


class TextIndex:
    def __init__(self, vectorizer):
        self.vec = vectorizer  # fitted TfidfVectorizer (rows are L2-normalised)
        self.ids: list[str] = []
        self.pos: dict[str, int] = {}
        self._blocks: list = []
        self._M = None

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], texts: list[str]) -> None:
        new = [(i, t) for i, t in zip(ids, texts) if i not in self.pos]
        if not new:
            return
        for i, _ in new:
            self.pos[i] = len(self.ids)
            self.ids.append(i)
        self._blocks.append(self.vec.transform([t for _, t in new]))
        self._M = None

    @property
    def M(self):
        if self._M is None and self._blocks:
            self._M = sparse.vstack(self._blocks).tocsr()
            self._blocks = [self._M]
        return self._M

    def query(self, texts: list[str], k: int, exclude: list[set[str]] | None = None) -> list[list[tuple[str, float]]]:
        if not self.ids or k <= 0:
            return [[] for _ in texts]
        Q = self.vec.transform(texts)
        S = (Q @ self.M.T).toarray()
        out = []
        for r in range(S.shape[0]):
            row = S[r]
            if exclude and exclude[r]:
                for cid in exclude[r]:
                    j = self.pos.get(cid)
                    if j is not None:
                        row[j] = -1.0
            kk = min(k, len(self.ids))
            top = np.argpartition(-row, kk - 1)[:kk]
            top = top[np.argsort(-row[top])]
            out.append([(self.ids[j], float(row[j])) for j in top if row[j] > 0])
        return out

    def sims(self, texts: list[str], other_texts: list[str]) -> np.ndarray:
        if not texts or not other_texts:
            return np.zeros((len(texts), len(other_texts)))
        return (self.vec.transform(texts) @ self.vec.transform(other_texts).T).toarray()
