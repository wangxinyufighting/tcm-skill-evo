"""Repository governance: deprecate / promote / merge (异病同治) / split (同病异治) / capacity."""
from __future__ import annotations

import logging
from collections import Counter

from . import prompts as P
from .induce import frequent_terms
from .skills import build_trigger, merge_skills, skill_from_json
from .utils import jaccard, truncate

LOG = logging.getLogger(__name__)


class Governor:
    def __init__(self, lib, store, matcher, verifier, llm, res, scfg, vocab_text: str):
        self.lib, self.store, self.matcher, self.verifier, self.llm, self.res = lib, store, matcher, verifier, llm, res
        self.g = scfg.govern
        self.scfg = scfg
        self.structured = bool(scfg.structured)
        self.vocab_text = vocab_text

    def run(self, window: int, step: int) -> dict:
        st: Counter = Counter()
        self._deprecate_promote(window, st)
        if self.scfg.merge:
            self._merge(step, st)
        if self.scfg.split and self.structured:
            self._split(step, st)
        self._remove_and_cap(window, st)
        return dict(st)

    # ------------------------------------------------------------------ lifecycle
    def _deprecate_promote(self, window: int, st: Counter) -> None:
        g = self.g
        for s in list(self.lib.skills.values()):
            if s.status == "deprecated":
                continue
            if (s.n_used >= int(g.dep_min_used) and s.utility < float(g.dep_util)) or (s.n_ret >= int(g.dep_min_ret_unused) and s.n_used == 0 and s.n_credit == 0):
                s.status, s.deprecated_at = "deprecated", window
                st["deprecated"] += 1
                self.lib.version += 1
            elif s.status == "active" and s.n_used >= int(g.promote_min_used) and s.utility >= float(g.promote_util):
                s.status = "mature"
                st["promoted"] += 1

    def _merge(self, step: int, st: Counter) -> None:
        g = self.g
        if self.structured:
            fs = sorted(self.lib.active("formula"), key=lambda s: -s.utility)
            removed: set[str] = set()
            for i, a in enumerate(fs):
                if a.id in removed:
                    continue
                for b in fs[i + 1 :]:
                    if b.id in removed:
                        continue
                    if jaccard(a.core_herbs(), b.core_herbs()) >= float(g.merge_core_jacc):
                        merged = merge_skills(a, b, float(g.merge_trig_jacc))
                        ok, _ = self.verifier.verify(merged)
                        if ok:
                            tag = "异病同治" if len(merged.triggers) > len(a.triggers) else "近似合并"
                            self.lib.replace(merged, step, f"merge {b.id} ({tag})")
                            self.lib.remove(b.id)
                            removed.add(b.id)
                            a = merged
                            st[f"merged:{tag}"] += 1
            ms = sorted(self.lib.active("modification"), key=lambda s: -s.utility)
            removed = set()
            for i, a in enumerate(ms):
                if a.id in removed:
                    continue
                for b in ms[i + 1 :]:
                    if b.id in removed:
                        continue
                    if jaccard(a.add, b.add) >= 0.5 and jaccard(a.all_terms(), b.all_terms()) >= 0.5:
                        merged = merge_skills(a, b)
                        self.lib.replace(merged, step, f"merge {b.id}")
                        self.lib.remove(b.id)
                        removed.add(b.id)
                        a = merged
                        st["merged:modification"] += 1
        for br in ("principle", "free"):
            pool = sorted(self.lib.active(br), key=lambda s: -s.utility)
            if len(pool) < 2:
                continue
            S = self.store.index.sims([s.text_key() for s in pool], [s.text_key() for s in pool])
            removed = set()
            for i in range(len(pool)):
                if pool[i].id in removed:
                    continue
                for j in range(i + 1, len(pool)):
                    if pool[j].id not in removed and S[i, j] >= float(g.merge_text_sim):
                        self.lib.remove(pool[j].id)
                        removed.add(pool[j].id)
                        st[f"merged:{br}"] += 1

    # ------------------------------------------------------------------ split (同病异治)
    def _split_candidates(self) -> list[tuple]:
        g = self.g
        out = []
        for s in self.lib.active("formula"):
            logs = [l for l in s.usage_log if l.get("adopted") and l["cid"] in self.store.pos]
            if len(logs) < int(g.split_min_used):
                continue
            items = [(self.store.texts[self.store.pos[l["cid"]]], self.store.symsets[self.store.pos[l["cid"]]], l["prec"], l["cid"]) for l in logs]
            terms = [t for t, c in frequent_terms([x[0] for x in items], [list(x[1]) for x in items], min_frac=0.2, max_terms=25) if c < len(items)]
            best = None
            for t in terms:
                w = [x for x in items if t in x[0] or t in x[1]]
                wo = [x for x in items if not (t in x[0] or t in x[1])]
                if len(w) < int(g.split_min_group) or len(wo) < int(g.split_min_group):
                    continue
                gap = sum(x[2] for x in wo) / len(wo) - sum(x[2] for x in w) / len(w)
                if gap >= float(g.split_gap) and (best is None or gap > best[0]):
                    best = (gap, t, wo, w)
            if best:
                out.append((best[0], s, best[1], best[2], best[3]))
        out.sort(key=lambda x: -x[0])
        return out[:3]

    def _split(self, step: int, st: Counter) -> None:
        cands = self._split_candidates()
        if not cands:
            return
        system = P.SPLIT_SYSTEM + self.vocab_text

        def run(c):
            gap, s, t, good, bad = c
            fmt = lambda xs: [f"- 主治：{truncate(x[0], 120)}\n  原方：{'、'.join(self.res.cases[x[3]].herbs)}" for x in xs[:4]]
            b_cnt = Counter(h for x in bad for h in self.res.cases[x[3]].herbs)
            b_core = [h for h, n in b_cnt.most_common() if n >= max(2, len(bad) // 2)]
            user = P.split_user(s.render(), fmt(good), fmt(bad), f"B 组描述多含“{t}”（A 组平均精度比 B 组高 {gap:.2f}）")
            meta = {"cids": [x[3] for x in good[:4] + bad[:4]], "sep_term": t, "required": s.triggers[0].required if s.triggers else [], "b_core": b_core}
            r = self.llm.chat_json([{"role": "system", "content": system}, {"role": "user", "content": user}], task="split", meta=meta)
            return c, (r.data if r.ok and isinstance(r.data, dict) else None)

        for (gap, s, t, good, bad), data in self.llm.map(run, cands, desc="split"):
            if data is None or self.lib.get(s.id) is None:
                continue
            grounding_a = [x[0] for x in good] + [y for x in good for y in x[1]]
            grounding_b = [x[0] for x in bad] + [y for x in bad for y in x[1]]
            narrow = (data.get("narrow") or {}).get("trigger")
            new_t = build_trigger(narrow, grounding_a + [s.render()], 2) if narrow else None
            cur = self.lib.get(s.id)
            if new_t is None:  # fallback: exclude the separating term
                for tr in cur.triggers:
                    if t not in tr.exclude:
                        tr.exclude.append(t)
            else:
                if t not in new_t.exclude and t not in new_t.required:
                    new_t.exclude.append(t)
                cur.triggers[0] = new_t
            cur.usage_log = []
            self.lib.replace(cur, step, f"split on '{t}'")
            st["split"] += 1
            ns = data.get("new_skill")
            if isinstance(ns, dict):
                ns["branch"] = "formula"
                sk, why = skill_from_json(ns, self.res.lexicon, grounding_b, True)
                if sk is not None:
                    ok, why = self.verifier.verify(sk)
                    if ok:
                        sk.utility = float(self.scfg.init_utility)
                        sk.evidence = [x[3] for x in bad]
                        self.lib.add(sk, step)
                        st["split_new_skill"] += 1
                    else:
                        st[f"split_new_rejected:{why}"] += 1

    # ------------------------------------------------------------------ cleanup
    def _remove_and_cap(self, window: int, st: Counter) -> None:
        g = self.g
        for s in list(self.lib.skills.values()):
            if s.status == "deprecated" and window - s.deprecated_at >= int(g.remove_after_windows):
                self.lib.remove(s.id)
                st["removed"] += 1
        for br, cap in dict(g.cap).items():
            act = [s for s in self.lib.active(br)]
            if len(act) <= int(cap):
                continue
            victims = sorted([s for s in act if s.status != "mature"], key=lambda s: (s.utility, s.support))[: len(act) - int(cap)]
            for s in victims:
                s.status, s.deprecated_at = "deprecated", window
                st["capacity_deprecated"] += 1
            self.lib.version += 1
