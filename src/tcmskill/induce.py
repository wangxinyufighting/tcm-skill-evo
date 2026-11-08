"""Skill writing: clustering -> multi-case contrastive induction (LLM) -> statistical verification
("LLM proposes, data verifies") -> novelty gate (CREATE→PATCH) -> library update.
Also mines candidate 随症加减 rules directly from reflections (no extra LLM call)."""
from __future__ import annotations

import math
from collections import Counter

from . import prompts as P
from .learning import ExperienceRecord, reflection_cues
from .skills import Skill, apply_patch, ground_terms, merge_skills, skill_from_json
from .utils import dedup_keep_order, jaccard, truncate

_STOP = set("者之而及或不等治主用于其以为与并兼诸凡一切也症病疾方汤丸散饮膏丹服后前时人日月年此因故则亦皆即能可宜忌")
_PUNCT = set("，。、；：,.;:！!？?（）()《》“”\"' 0123456789①②③④⑤⑥⑦⑧⑨⑩—-…")


# ============================================================================ text statistics
def frequent_terms(texts: list[str], sym_lists: list[list[str]] | None = None, min_frac: float = 0.34, max_terms: int = 12) -> list[tuple[str, int]]:
    n = len(texts)
    if n == 0:
        return []
    thr = max(2, math.ceil(min_frac * n))
    dc: Counter = Counter()
    for t in texts:
        grams = set()
        for L in (2, 3, 4):
            for i in range(len(t) - L + 1):
                g = t[i : i + L]
                if any(ch in _PUNCT for ch in g) or g[0] in _STOP or g[-1] in _STOP:
                    continue
                grams.add(g)
        dc.update(grams)
    kc: Counter = Counter()
    for syms in sym_lists or []:
        kc.update(set(syms))
    cands = [(g, c) for g, c in dc.items() if c >= thr]
    cands += [(k, c) for k, c in kc.items() if c >= thr and k not in dc]
    cands.sort(key=lambda x: (-x[1], -len(x[0])))
    kept: list[tuple[str, int]] = []
    for g, c in cands:
        if any(g in k and c <= kc_ for k, kc_ in kept):
            continue
        kept.append((g, c))
        if len(kept) >= max_terms:
            break
    return kept


def differential(items: list[dict], herb_cnt: Counter, terms: list[str], max_lines: int = 6) -> tuple[list[str], list[dict]]:
    n = len(items)
    if n < 4:
        return [], []
    has = lambda it, t: t in it["text"] or t in it.get("symptoms", [])
    out = []
    for h, c in herb_cnt.items():
        if not (0.2 * n <= c <= 0.85 * n):
            continue
        best = None
        for t in terms:
            with_t = [it for it in items if has(it, t)]
            b = len(with_t)
            if b < 2 or b == n:
                continue
            a = sum(1 for it in with_t if h in it["gt"])
            cc = sum(1 for it in items if not has(it, t) and h in it["gt"])
            d = n - b
            contrast = a / b - cc / d
            if contrast >= 0.5 and (best is None or contrast > best[0]):
                best = (contrast, t, a, b, cc, d)
        if best:
            out.append((best[0], h, best))
    out.sort(key=lambda x: -x[0])
    lines = [f"描述含“{b[1]}”的 {b[2]}/{b[3]} 例用了{h}，不含的 {b[4]}/{b[5]} 例用了" for _, h, b in out[:max_lines]]
    return lines, [{"term": b[1], "herb": h} for _, h, b in out[:max_lines]]


# ============================================================================ clustering
class Clusterer:
    def __init__(self, ccfg, single: bool = False):
        self.c = ccfg
        self.single = single

    def build(self, recs: list[ExperienceRecord], store) -> tuple[list[dict], list[ExperienceRecord]]:
        c = self.c
        recs = sorted(recs, key=lambda r: -r.priority)
        assigned: set[str] = set()
        clusters: list[dict] = []
        leftovers: list[ExperienceRecord] = []
        for seed in recs:
            if seed.cid in assigned:
                continue
            if len(clusters) >= int(c.max_clusters_per_window):
                leftovers.append(seed)
                continue
            if self.single:
                clusters.append({"records": [seed], "hist": []})
                assigned.add(seed.cid)
                continue
            members = [seed] + [r for r in recs if r.cid not in assigned and r is not seed and jaccard(r.gt, seed.gt) >= float(c.herb_jaccard)][: int(c.k_max) - 1]
            ex = {m.cid for m in members}
            hist = store.neighbors_by_herbs(seed.gt, k=int(c.hist_neighbors), min_jacc=float(c.herb_jaccard), exclude=ex)
            if len(members) + len(hist) >= int(c.k_min):
                clusters.append({"records": members, "hist": [i for i, _ in hist]})
                assigned |= ex
            else:
                leftovers.append(seed)
        return clusters, leftovers


# ============================================================================ induction
class Inducer:
    def __init__(self, llm, res, store, lib, matcher, scfg, vocab_text: str):
        self.llm, self.res, self.store, self.lib, self.matcher = llm, res, store, lib, matcher
        self.structured = bool(scfg.structured)
        self.system = (P.INDUCE_SYSTEM if self.structured else P.INDUCE_FREE_SYSTEM) + vocab_text

    def items(self, cl: dict) -> list[dict]:
        out = []
        for r in cl["records"]:
            lesson = (r.reflection or {}).get("lesson", "") if r.reflection else ""
            out.append({"cid": r.cid, "text": r.text, "symptoms": r.symptoms, "gt": r.gt, "fn": r.fn, "fp": r.fp, "lesson": str(lesson)[:80], "attempted": r.attempted})
        st = self.store
        for i in cl["hist"]:
            cid = st.cids[i]
            case = self.res.cases[cid]
            out.append({"cid": cid, "text": st.texts[i], "symptoms": list(case.symptoms), "gt": list(case.herbs), "fn": [], "fp": [], "lesson": "", "attempted": False})
        return out

    def related(self, core: list[str], items: list[dict], k: int = 3) -> list[Skill]:
        texts = [it["text"] for it in items]
        herbs = set(h for it in items for h in it["gt"])
        rel = []
        if self.structured:
            for s in self.lib.active("formula"):
                cj = jaccard(core, s.core_herbs()) if core else 0.0
                fires = sum(1 for it in items if self.matcher.match(s, it["text"], set(it["symptoms"])) >= self.matcher.fire)
                if cj >= 0.3 or fires >= max(2, len(items) // 2):
                    rel.append((cj + fires / len(items), s))
            for s in self.lib.active("modification"):
                if set(s.add) & herbs and any(any(t in tx for tx in texts) for t in s.all_terms()):
                    rel.append((0.5 * len(set(s.add) & herbs), s))
        else:
            M = [s for s in self.lib.active("free")]
            if M:
                sims = self.store.index.sims([" ".join(texts)[:800]], [s.text_key() for s in M])[0]
                rel = [(float(sims[j]), s) for j, s in enumerate(M) if sims[j] > 0.2]
        rel.sort(key=lambda x: -x[0])
        return [s for _, s in rel[: 2 * k]]

    def induce(self, cl: dict) -> dict:
        items = self.items(cl)
        n = len(items)
        herb_cnt = Counter(h for it in items for h in set(it["gt"]))
        core = [h for h, c in herb_cnt.most_common() if c >= max(2, math.ceil(n / 2))] if n > 1 else list(items[0]["gt"])
        terms = frequent_terms([it["text"] for it in items], [it["symptoms"] for it in items]) if n > 1 else [(t, 1) for t in items[0]["symptoms"]]
        diff_lines, diff = differential(items, herb_cnt, [t for t, _ in terms] + sorted({s for it in items for s in it["symptoms"]}))
        hint = [f"- 药物出现次数（共 {n} 例）：" + "、".join(f"{h} {c}" for h, c in herb_cnt.most_common(24))]
        if n > 1:
            hint.append("- 出现于≥一半病例的药（核心候选）：" + ("、".join(core) or "无"))
            hint.append("- 高频描述词：" + ("、".join(f"{t}({c})" for t, c in terms) or "无"))
        if diff_lines:
            hint.append("- 差异线索：" + "；".join(diff_lines))
        rel = self.related(core, items)
        blocks = []
        for i, it in enumerate(items, 1):
            b = f"#{i} 主治：{truncate(it['text'], 150)}\n   原方：{'、'.join(it['gt'])}"
            if it["attempted"]:
                b += f"\n   学员漏开：{'、'.join(it['fn']) or '无'}；多开：{'、'.join(it['fp']) or '无'}"
            if it["lesson"]:
                b += f"\n   经验：{it['lesson']}"
            blocks.append(b)
        user = P.induce_user(blocks, "\n".join(hint), "\n".join(s.render() for s in rel))
        meta = {"cids": [it["cid"] for it in items], "cases": [{"cid": it["cid"]} for it in items], "terms": [t for t, _ in terms], "core": core, "diff": diff, "related": [s.id for s in rel], "structured": self.structured}
        r = self.llm.chat_json([{"role": "system", "content": self.system}, {"role": "user", "content": user}], task="induce", meta=meta)
        ops = []
        if r.ok and isinstance(r.data, dict):
            ops = [o for o in (r.data.get("operations") or []) if isinstance(o, dict)]
        grounding = [it["text"] for it in items] + [s for it in items for s in it["symptoms"]]
        return {"ops": ops, "grounding": grounding, "cids": [it["cid"] for it in items]}

    def induce_batch(self, clusters: list[dict]) -> list[dict]:
        return self.llm.map(self.induce, clusters, desc="induce")


# ============================================================================ verification
class Verifier:
    def __init__(self, store, matcher, res, scfg, safety_mode: str = "report"):
        self.store, self.matcher, self.res = store, matcher, res
        self.enabled = bool(scfg.verify)
        self.t = scfg.verify_thr
        self.safety_mode = safety_mode

    def verify(self, s: Skill, source_cids=None) -> tuple[bool, str]:
        """source_cids: cases the skill was induced from. Their out-of-sample stats are always recorded;
        with verify_thr.holdout_source=true the thresholds are applied to out-of-sample cases only."""
        if s.branch in ("principle", "free"):
            return True, "text_skill"
        if not self.enabled:
            return True, "unverified"
        src = {self.store.pos[c] for c in (source_cids or []) if c in self.store.pos}
        if s.branch == "formula":
            return self._formula(s, src)
        return self._modification(s, src)

    def _safety(self, herbs: list[str]) -> list[str]:
        if self.safety_mode != "filter":
            return herbs
        kept, _ = self.res.incompat.filter(herbs)
        return kept

    def _formula(self, s: Skill, src: set = frozenset()) -> tuple[bool, str]:
        st, t = self.store, self.t
        idx_all = self.matcher.matched_cases(s)
        idx_out = [i for i in idx_all if i not in src]
        core0 = set(s.core_herbs())
        prec_out = sum(len(core0 & st.herbsets[i]) / len(core0) for i in idx_out) / len(idx_out) if idx_out and core0 else 0.0
        idx = idx_out if t.get("holdout_source", False) else idx_all
        n = len(idx)
        if n < int(t.formula_min_match):
            return False, "too_few_matches"
        core = set(s.core_herbs())
        prec = sum(len(core & st.herbsets[i]) / len(core) for i in idx) / n
        base = sum(st.herb_p(h) for h in core) / len(core)
        lift = prec / base if base > 0 else 0.0
        if prec < float(t.formula_min_prec):
            return False, "low_precision"
        if lift < float(t.formula_min_lift):
            return False, "low_lift"
        within = set(idx)
        new_mods = []
        for m in s.mods:
            X = [i for i in within if self.matcher.mod_fires(m, st.texts[i], st.symsets[i])]
            keep = []
            for h in m.get("add", []):
                n_xy = sum(1 for i in X if h in st.herbsets[i])
                p_ctx = sum(1 for i in within if h in st.herbsets[i]) / n
                conf = n_xy / len(X) if X else 0.0
                lift_m = conf / p_ctx if p_ctx > 0 else (99.0 if n_xy > 0 else 0.0)
                if n_xy >= int(t.mod_min_support) and lift_m >= float(t.mod_min_lift):
                    keep.append(h)
                elif len(X) < int(t.mod_min_support):
                    g = st.rule_stats(m["if"], h)
                    if g["n_xy"] >= int(t.rule_min_support) and g["lift"] >= float(t.rule_min_lift):
                        keep.append(h)
            if keep or m.get("remove"):
                new_mods.append({"if": m["if"], "add": keep, "remove": m.get("remove", [])})
        s.mods = new_mods
        kept = self._safety(s.core_herbs())
        if len(kept) < len(s.core):
            s.core = [c for c in s.core if c["name"] in set(kept)]
        s.support = n
        s.vstats = {"n_match": n, "prec": round(prec, 3), "lift": round(lift, 2), "n_match_out": len(idx_out), "prec_out": round(prec_out, 3)}
        return True, "ok"

    def _modification(self, s: Skill, src: set = frozenset()) -> tuple[bool, str]:
        st, t = self.store, self.t
        terms = s.triggers[0].required if s.triggers else []
        if not terms:
            return False, "no_trigger"
        within = (set(range(st.n)) - src) if (src and t.get("holdout_source", False)) else None
        keep, best = [], None
        for h in s.add:
            g = st.rule_stats(terms, h, within)
            if g["n_xy"] >= int(t.rule_min_support) and g["lift"] >= float(t.rule_min_lift) and g["conf"] >= float(t.rule_min_conf):
                keep.append(h)
                if best is None or g["n_xy"] > best["n_xy"]:
                    best = g
        if not keep:
            return False, "rule_not_supported"
        s.add = self._safety(keep) or keep
        s.support = best["n_xy"]
        s.vstats = best
        return True, "ok"


# ============================================================================ rule mining from reflections
def rules_from_reflections(recs: list[ExperienceRecord], lexicon, min_len: int = 2) -> list[dict]:
    seen, ops = set(), []
    for rec in recs:
        grounding = [rec.text] + list(rec.symptoms)
        for herb, cues, reason in reflection_cues(rec, lexicon):
            for cue in ground_terms(cues, grounding, min_len)[:2]:
                if (cue, herb) in seen:
                    continue
                seen.add((cue, herb))
                ops.append(
                    {
                        "op": "CREATE",
                        "skill": {"branch": "modification", "trigger": {"required": [cue], "optional": []}, "context": "", "add": [herb], "note": reason},
                        "_grounding": grounding,
                        "_cids": [rec.cid],
                    }
                )
    return ops


# ============================================================================ writer
class Writer:
    def __init__(self, lib, verifier: Verifier, res, scfg):
        self.lib, self.verifier, self.res = lib, verifier, res
        self.structured = bool(scfg.structured)
        self.init_u = float(scfg.init_utility)
        self.min_len = int(scfg.verify_thr.min_term_len)

    def _text_sim(self, a: str, b: list[str]) -> list[float]:
        if not b:
            return []
        return list(self.verifier.store.index.sims([a], b)[0])

    def find_dup(self, s: Skill) -> Skill | None:
        if s.branch == "formula":
            for t in self.lib.active("formula"):
                if jaccard(t.core_herbs(), s.core_herbs()) >= 0.6 and (jaccard(t.all_terms(), s.all_terms()) >= 0.3 or (s.name and s.name == t.name)):
                    return t
        elif s.branch == "modification":
            for t in self.lib.active("modification"):
                if jaccard(t.add, s.add) >= 0.5 and t.triggers and s.triggers and jaccard(t.triggers[0].required, s.triggers[0].required) >= 0.5:
                    return t
        else:
            pool = self.lib.active(s.branch)
            sims = self._text_sim(s.text_key(), [t.text_key() for t in pool])
            if sims:
                j = max(range(len(sims)), key=lambda i: sims[i])
                if sims[j] >= 0.8:
                    return pool[j]
        return None

    def apply(self, ops: list[dict], grounding: list[str], cids: list[str], step: int) -> Counter:
        stats: Counter = Counter()
        for op in ops:
            kind = str(op.get("op", "")).upper()
            g = op.get("_grounding", grounding)
            ev = op.get("_cids", cids)
            if kind == "CREATE":
                s, why = skill_from_json(op.get("skill"), self.res.lexicon, g, self.structured, self.min_len)
                if s is None:
                    stats[f"invalid:{why}"] += 1
                    continue
                s.evidence = list(ev)[-30:]
                dup = self.find_dup(s)
                if dup is not None:
                    merged = merge_skills(dup, s)
                    ok, why = self.verifier.verify(merged, ev)
                    if ok:
                        self.lib.replace(merged, step, "merge-on-create")
                        stats["merged_into_existing"] += 1
                    else:
                        stats[f"dup_rejected:{why}"] += 1
                    continue
                ok, why = self.verifier.verify(s, ev)
                if not ok:
                    stats[f"rejected:{why}"] += 1
                    continue
                s.utility = self.init_u - (0.1 if s.branch == "principle" else 0.0)
                self.lib.add(s, step)
                stats[f"created:{s.branch}"] += 1
            elif kind == "PATCH":
                target = self.lib.get(str(op.get("id", "")).strip("[]【】 "))
                if target is None or target.status == "deprecated":
                    stats["patch_bad_id"] += 1
                    continue
                new = apply_patch(target, op.get("patch") or {}, self.res.lexicon, g, self.min_len)
                new.evidence = dedup_keep_order(new.evidence + list(ev))[-30:]
                ok, why = self.verifier.verify(new, ev)
                if ok:
                    self.lib.replace(new, step, "patch")
                    stats["patched"] += 1
                else:
                    stats[f"patch_rejected:{why}"] += 1
            elif kind == "NONE":
                stats["none"] += 1
            else:
                stats["unknown_op"] += 1
        return stats
