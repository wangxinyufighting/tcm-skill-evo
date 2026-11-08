"""Non-LLM baselines: frequency, kNN voting, MLP multi-label classifier."""
from __future__ import annotations

import logging
from collections import Counter

import numpy as np

from .prescriber import Prediction
from .textindex import TextIndex

LOG = logging.getLogger(__name__)


class FreqBaseline:
    name = "freq"

    def __init__(self, res):
        self.res = res

    def fit(self, train_ids):
        cnt = Counter(h for c in train_ids for h in self.res.cases[c].herbs)
        lens = sorted(len(self.res.cases[c].herbs) for c in train_ids)
        self.top = [h for h, _ in cnt.most_common()]
        self.L = lens[len(lens) // 2]
        return self

    def predict(self, cases):
        return [Prediction(cid=c.cid, herbs=self.top[: self.L]) for c in cases]


class KNNBaseline:
    """Similarity-weighted herb voting over the k most similar training cases.
    Prescription length = similarity-weighted mean length of the neighbours."""

    name = "knn"

    def __init__(self, res, k: int = 20, feature: str = "description"):
        self.res, self.k, self.feature = res, k, feature

    def _text(self, c):
        return "、".join(c.symptoms) if self.feature == "keywords" else self.res.text(c)

    def fit(self, train_ids):
        if self.feature == "keywords":
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(analyzer=lambda s: s.split("、"), sublinear_tf=True)
            vec.fit([self._text(self.res.cases[c]) for c in train_ids])
        else:
            vec = self.res.vectorizer
        self.index = TextIndex(vec)
        self.index.add(list(train_ids), [self._text(self.res.cases[c]) for c in train_ids])
        return self

    def predict(self, cases):
        out = []
        nbrs = self.index.query([self._text(c) for c in cases], self.k, exclude=[{c.cid} for c in cases])
        for c, nb in zip(cases, nbrs):
            if not nb:
                out.append(Prediction(cid=c.cid, herbs=self.res.labels[: self.res.median_len]))
                continue
            score: Counter = Counter()
            wsum = sum(s for _, s in nb)
            L = sum(s * len(self.res.cases[cid].herbs) for cid, s in nb) / max(wsum, 1e-9)
            for cid, s in nb:
                for h in self.res.cases[cid].herbs:
                    score[h] += s
            out.append(Prediction(cid=c.cid, herbs=[h for h, _ in score.most_common(max(1, int(round(L))))]))
        return out


class MLPBaseline:
    """TF-IDF -> MLP (sigmoid outputs, sklearn). Threshold tuned on the validation split."""

    name = "mlp"

    def __init__(self, res, hidden: int = 512, max_iter: int = 40, seed: int = 0):
        self.res, self.hidden, self.max_iter, self.seed = res, hidden, max_iter, seed

    def fit(self, train_ids, val_ids=None):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MultiLabelBinarizer

        self.mlb = MultiLabelBinarizer(classes=self.res.labels)
        Y = self.mlb.fit_transform([self.res.cases[c].herbs for c in train_ids])
        X = self.res.vectorizer.transform([self.res.text(self.res.cases[c]) for c in train_ids])
        LOG.info("training MLP on %s ...", X.shape)
        self.clf = MLPClassifier(hidden_layer_sizes=(self.hidden,), max_iter=self.max_iter, early_stopping=True, random_state=self.seed, verbose=False)
        self.clf.fit(X, Y)
        self.thr = 0.3
        if val_ids:
            from .metrics import prf

            P = self._proba([self.res.cases[c] for c in val_ids])
            best = (-1, 0.3)
            for t in np.arange(0.05, 0.6, 0.05):
                f = np.mean([prf(self._decode(p, t), self.res.cases[c].herbs)[2] for p, c in zip(P, val_ids)])
                best = max(best, (f, float(t)))
            self.thr = best[1]
            LOG.info("MLP threshold tuned on val: %.2f (F1 %.3f)", self.thr, best[0])
        return self

    def _proba(self, cases):
        X = self.res.vectorizer.transform([self.res.text(c) for c in cases])
        return self.clf.predict_proba(X)

    def _decode(self, p, thr):
        idx = np.where(p >= thr)[0]
        if len(idx) == 0:
            idx = np.argsort(-p)[:1]
        idx = idx[np.argsort(-p[idx])]
        return [self.mlb.classes_[i] for i in idx]

    def predict(self, cases):
        P = self._proba(cases)
        return [Prediction(cid=c.cid, herbs=self._decode(p, self.thr)) for c, p in zip(cases, P)]
