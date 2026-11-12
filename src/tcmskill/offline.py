"""Offline bulk skill induction from the whole training split ("read the casebook first").

The online learner (methods.SkillMethod) sees a few hundred cases and writes a handful of formula
skills. Here every training case is used at once; the statistics need no LLM:

  1. vocabulary   literal symptom phrases (text between punctuation that occurs as a whole phrase in
                  >= min_phrase_df training descriptions) + the PTM symptom keyword list
  2. families     greedy herb-Jaccard clustering of the training formulas (方族)
  3. formula      one skill per family: core herbs (>= core_frac of members); up to max_triggers
                  literal triggers picked by greedy member cover under a precision constraint
                  (several triggers = 异病同治); within-family 随症加减 by contrast
  4. modification global term -> herb rules (support / confidence / lift over all training cases)
  5. verify       the online Verifier (same matcher, same thresholds) on all training cases;
                  vstats.prec_out = precision on matched cases outside the family
  6. annotate     optional LLM pass naming each family (方名 / 病机 / 治法 / 君臣佐使);
                  triggers and herbs stay data-derived ("LLM names, data decides")

The result is a memory file for `run_method.py --load_memory` (methods ours / ours_hybrid / ours_select).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse

from .induce import Verifier, differential
from .metrics import prf
from .skills import ROLES, Skill, SkillLibrary, SkillMatcher, Trigger, merge_mods
from .stats import SeenStore
from .utils import dedup_keep_order, truncate

LOG = logging.getLogger(__name__)

_SPLIT_RE = re.compile(r"[，,。.、；;：:！!？?（）()\[\]【】《》“”\"'\s]+")
_LEAD = ("主治", "用于", "治疗", "兼治", "专治", "又治", "亦治", "治", "主", "疗", "兼", "及", "或", "并", "凡", "又", "亦", "其", "如")
_TAIL = ("等症", "诸症", "诸证", "等证", "等", "者", "也")
_BAD = re.compile(r"[0-9A-Za-z]")


def phrases(text: str, min_len: int = 2, max_len: int = 8) -> list[str]:
    out = []
    for p in _SPLIT_RE.split(text or ""):
        for lead in _LEAD:
            if p.startswith(lead) and len(p) - len(lead) >= min_len:
                p = p[len(lead):]
                break
        for tail in _TAIL:
            if p.endswith(tail) and len(p) - len(tail) >= min_len:
                p = p[: -len(tail)]
                break
        if min_len <= len(p) <= max_len and not _BAD.search(p):
            out.append(p)
    return out


def build_vocab(texts: list[str], symptom_vocab, min_phrase_df: int = 3, min_len: int = 2, max_len: int = 8) -> set[str]:
    cnt: Counter = Counter()
    for t in texts:
        cnt.update(set(phrases(t, min_len, max_len)))
    vocab = {p for p, n in cnt.items() if n >= min_phrase_df}
    vocab |= {s for s in symptom_vocab or [] if min_len <= len(s) <= max_len and not _BAD.search(s)}
    return vocab


def term_postings(texts: list[str], symsets: list, vocab: set[str]) -> tuple[dict[str, list[int]], list[set[str]]]:
    """Presence = literal substring of the description OR a PTM symptom keyword (same as SkillMatcher.present)."""
    lens = sorted({len(v) for v in vocab})
    post: dict[str, list[int]] = defaultdict(list)
    terms_of: list[set[str]] = []
    for i, (t, syms) in enumerate(zip(texts, symsets)):
        found = {s for s in syms if s in vocab}
        for L in lens:
            for j in range(len(t) - L + 1):
                g = t[j : j + L]
                if g in vocab:
                    found.add(g)
        for g in found:
            post[g].append(i)
        terms_of.append(found)
    return dict(post), terms_of


def herb_matrix(herbsets: list, labels: list[str]) -> tuple[sparse.csr_matrix, dict[str, int]]:
    lab = {h: i for i, h in enumerate(labels)}
    r, c = [], []
    for i, hs in enumerate(herbsets):
        for h in hs:
            if h in lab:
                r.append(i)
                c.append(lab[h])
    X = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(herbsets), len(labels)))
    return X, lab


def cluster_families(X: sparse.csr_matrix, thr: float = 0.5, min_size: int = 3, chunk: int = 1000) -> list[list[int]]:
    """Greedy dense-seed clustering: densest unassigned case + its unassigned neighbours with herb Jaccard >= thr.
    The seed (a real formula at the centre of the family) is the first member of every family."""
    n = X.shape[0]
    L = np.asarray(X.sum(1)).ravel().astype(np.float32)
    XT = X.T.tocsc()
    nbrs = []
    for s in range(0, n, chunk):
        inter = (X[s : s + chunk] @ XT).toarray()
        jac = inter / np.maximum(L[s : s + chunk, None] + L[None, :] - inter, 1e-9)
        for r in range(jac.shape[0]):
            nbrs.append(np.flatnonzero(jac[r] >= thr))
    order = np.argsort([-len(x) for x in nbrs], kind="stable")
    assigned = np.zeros(n, bool)
    fams = []
    for i in order:
        if assigned[i]:
            continue
        mem = [int(i)] + [int(j) for j in nbrs[i] if not assigned[j] and j != i]
        if len(mem) >= min_size:
            assigned[mem] = True
            fams.append(mem)
    return fams


class OfflineBuilder:
    def __init__(self, res, cfg, cids: list[str]):
        self.res, self.cfg = res, cfg
        self.oc = cfg.offline
        sc = cfg.skills
        self.sc = sc
        self.store = SeenStore(res)
        self.store.add(res.get(cids))
        st = self.store
        self.N = st.n
        LOG.info("词表与倒排 …")
        self.vocab = build_vocab(st.texts, res.meta.get("ptm_symptoms", []), int(self.oc.min_phrase_df))
        self.post, self.terms_of = term_postings(st.texts, st.symsets, self.vocab)
        for t, rows in self.post.items():  # share postings with the matcher / verifier (identical presence rule)
            st._term_post[t] = set(rows)
        self.post_np = {t: np.asarray(rows, dtype=np.int64) for t, rows in self.post.items()}
        self.X, self.lab = herb_matrix(st.herbsets, res.labels)
        self.Xc = self.X.tocsc()
        self.matcher = SkillMatcher(st, sc.match)
        self.verifier = Verifier(st, self.matcher, res, sc, cfg.safety.mode)
        self.lib = SkillLibrary()
        self.stats: Counter = Counter()

    # ------------------------------------------------------------------ formula skills
    def family_skill(self, members: list[int]) -> tuple[Skill | None, str]:
        oc, st = self.oc, self.store
        n = len(members)
        herb_cnt = Counter(h for i in members for h in st.herbsets[i])
        if str(oc.get("core_from", "frac")) == "seed":  # the centre formula, minus herbs that are rare within the family
            seed = members[0]
            core = [h for h in sorted(st.herbsets[seed], key=lambda h: -herb_cnt[h]) if herb_cnt[h] >= float(oc.seed_keep_frac) * n and h in self.lab][: int(oc.max_core)]
        else:
            core = [h for h, c in herb_cnt.most_common() if c >= float(oc.core_frac) * n and h in self.lab][: int(oc.max_core)]
        if len(core) < int(oc.min_core):
            return None, "small_core"
        cov = np.asarray(self.Xc[:, [self.lab[h] for h in core]].sum(1)).ravel() / len(core)  # core coverage of every training case
        mem_of: dict[str, set[int]] = defaultdict(set)
        for i in members:
            for t in self.terms_of[i]:
                mem_of[t].add(i)
        cands = []
        for t, mem in mem_of.items():
            a = len(mem)
            if a < 2:
                continue
            rows = self.post_np[t]
            lift = (a / n) / (len(rows) / self.N)
            if lift < float(oc.min_lift):
                continue
            cands.append({"t": t, "a": a, "df": len(rows), "lift": lift, "prec": float(cov[rows].mean()), "mem": mem})
        if not cands:
            return None, "no_trigger_terms"
        max_df = int(oc.get("max_trigger_df", 0)) or self.N
        good = [c for c in cands if c["prec"] >= float(oc.min_prec) and int(self.sc.verify_thr.formula_min_match) <= c["df"] <= max_df]
        uncovered, triggers = set(members), []
        while good and len(triggers) < int(oc.max_triggers):
            best = max(good, key=lambda c: (len(c["mem"] & uncovered) * c["prec"], c["prec"], -c["df"]))
            if len(best["mem"] & uncovered) < int(oc.min_cover):
                break
            opt = [c["t"] for c in sorted(cands, key=lambda c: -c["a"])
                   if c["t"] != best["t"] and c["t"] not in best["t"] and best["t"] not in c["t"]
                   and len(c["mem"] & best["mem"]) >= 0.5 * len(best["mem"])][:4]
            triggers.append(Trigger(required=[best["t"]], optional=opt))
            uncovered -= best["mem"]
            good = [c for c in good if c is not best and c["t"] not in best["t"] and best["t"] not in c["t"]]
        if not triggers and bool(oc.pair_triggers):  # no single phrase is specific enough: try a conjunction
            top = sorted(cands, key=lambda c: -c["a"])[:10]
            pairs = []
            for x in range(len(top)):
                for y in range(x + 1, len(top)):
                    A, B = top[x], top[y]
                    if A["t"] in B["t"] or B["t"] in A["t"]:
                        continue
                    mem = A["mem"] & B["mem"]
                    if len(mem) < int(oc.min_cover):
                        continue
                    rows = np.intersect1d(self.post_np[A["t"]], self.post_np[B["t"]], assume_unique=True)
                    if len(rows) < int(self.sc.verify_thr.formula_min_match):
                        continue
                    prec = float(cov[rows].mean())
                    if prec >= float(oc.min_prec):
                        pairs.append((len(mem) * prec, A["t"], B["t"], mem))
            if pairs:
                _, t1, t2, mem = max(pairs, key=lambda x: x[0])
                triggers.append(Trigger(required=[t1, t2]))
                uncovered -= mem
        if not triggers:
            return None, "no_precise_trigger"
        items = [{"text": st.texts[i], "symptoms": list(st.symsets[i]), "gt": list(st.herbsets[i])} for i in members]
        terms = [t for t, m in sorted(mem_of.items(), key=lambda kv: -len(kv[1])) if len(m) >= 2][:40]
        _, diff = differential(items, herb_cnt, terms)
        mods = merge_mods([{"if": [d["term"]], "add": [d["herb"]], "remove": []} for d in diff if d["herb"] not in core])
        name = "、".join(core[:3]) + ("等" if len(core) > 3 else "") + "方族"
        s = Skill(branch="formula", name=name, triggers=triggers, core=[{"name": h, "role": ""} for h in core], mods=mods[:8],
                  note=f"同族{n}方", evidence=[st.cids[i] for i in members], support=n)  # full membership: skills.link_cases
        return s, "ok"

    def build_formulas(self) -> list[list[int]]:
        oc = self.oc
        fams = cluster_families(self.X, float(oc.family_jaccard), int(oc.family_min_size))
        LOG.info("方族 %d 个（覆盖 %d / %d 例）", len(fams), sum(len(f) for f in fams), self.N)
        for mem in fams:
            s, why = self.family_skill(mem)
            if s is None:
                self.stats[f"formula_skipped:{why}"] += 1
                continue
            ok, why = self.verifier.verify(s, [self.store.cids[i] for i in mem])
            if not ok:
                self.stats[f"formula_rejected:{why}"] += 1
                continue
            s.utility = float(np.clip(s.vstats.get("prec", 0.5), 0.3, 0.8))
            self.lib.add(s, 0)
            self.stats["created:formula"] += 1
        return fams

    # ------------------------------------------------------------------ modification rules
    def build_rules(self) -> None:
        oc = self.oc
        terms = sorted(t for t, rows in self.post.items() if len(rows) >= int(oc.rule_min_df))
        if not terms:
            return
        r, c = [], []
        for j, t in enumerate(terms):
            r += self.post[t]
            c += [j] * len(self.post[t])
        T = sparse.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(self.N, len(terms)))
        C = (T.T @ self.X).toarray()
        df = np.asarray(T.sum(0)).ravel()
        p = np.asarray(self.X.sum(0)).ravel() / self.N
        conf = C / df[:, None]
        lift = conf / np.maximum(p[None, :], 1e-9)
        labels = self.res.labels
        rules: dict[str, list[tuple[str, int, float, float]]] = {}
        for j, t in enumerate(terms):
            ok = np.flatnonzero((C[j] >= int(oc.rule_min_support)) & (conf[j] >= float(oc.rule_min_conf)) & (lift[j] >= float(oc.rule_min_lift)))
            if not len(ok):
                continue
            ok = ok[np.argsort(-(conf[j, ok] * np.log(lift[j, ok])))][: int(oc.rule_max_herbs)]
            rules[t] = [(labels[k], int(C[j, k]), float(conf[j, k]), float(lift[j, k])) for k in ok]
        # redundancy: drop a longer phrase when a phrase inside it already gives the same herbs about as reliably
        drop = set()
        for t, hs in rules.items():
            for s_ in rules:
                if s_ != t and s_ in t and {h for h, *_ in rules[s_]} >= {h for h, *_ in hs} and \
                        max(x[2] for x in rules[s_]) >= max(x[2] for x in hs) - 0.1:
                    drop.add(t)
                    break
        for t in sorted(rules, key=lambda t: -max(x[2] for x in rules[t])):
            if t in drop:
                self.stats["rule_redundant"] += 1
                continue
            s = Skill(branch="modification", triggers=[Trigger(required=[t])], add=[h for h, *_ in rules[t]],
                      note="；".join(f"{h} {n}/{len(self.post[t])}" for h, n, _, _ in rules[t]))
            ok, why = self.verifier.verify(s, [])
            if not ok:
                self.stats[f"rule_rejected:{why}"] += 1
                continue
            s.utility = float(np.clip(s.vstats.get("conf", 0.5), 0.3, 0.8))
            self.lib.add(s, 0)
            self.stats["created:modification"] += 1

    # ------------------------------------------------------------------ LLM naming (optional)
    def annotate(self, llm, max_cases: int = 8) -> None:
        from . import prompts as P

        skills = self.lib.active("formula")
        st = self.store

        def one(s: Skill):
            idx = [st.pos[c] for c in s.evidence if c in st.pos]
            seen_txt, picked = set(), []
            for i in idx:  # prefer different descriptions
                if st.texts[i] not in seen_txt:
                    seen_txt.add(st.texts[i])
                    picked.append(i)
                if len(picked) >= max_cases:
                    break
            blocks = [f"#{k + 1} 主治：{truncate(st.texts[i], 120)}\n   原方：{'、'.join(sorted(st.herbsets[i], key=lambda h: -self.res.herb_freq.get(h, 0)))}" for k, i in enumerate(picked)]
            trig = "；".join(t.render() for t in s.triggers)
            user = P.annotate_user(blocks, s.core_herbs(), trig)
            r = llm.chat_json([{"role": "system", "content": P.ANNOTATE_SYSTEM}, {"role": "user", "content": user}], task="annotate",
                              meta={"cid": s.id, "core": s.core_herbs()})
            return s, (r.data if r.ok and isinstance(r.data, dict) else None)

        done = 0
        for s, d in llm.map(one, skills, desc="annotate"):
            if not d:
                self.stats["annotate_failed"] += 1
                continue
            if d.get("name"):
                s.name = truncate(str(d["name"]), 30)
            s.pathogenesis = truncate(str(d.get("pathogenesis") or ""), 60)
            s.principle = truncate(str(d.get("principle") or ""), 40)
            roles = d.get("roles") if isinstance(d.get("roles"), dict) else {}
            norm = {}
            for k, v in roles.items():
                for lab in self.res.lexicon.normalize(str(k))[:1]:
                    norm[lab] = str(v)[:1]
            for c in s.core:
                if norm.get(c["name"]) in ROLES:
                    c["role"] = norm[c["name"]]
            if d.get("note"):
                s.note = truncate(f"{d['note']}（同族{len(s.evidence)}方）", 80)
            done += 1
        self.stats["annotated"] = done

    # ------------------------------------------------------------------ outputs
    def memory(self) -> dict:
        return {"step": 0, "window": 0, "library": self.lib.state_dict(), "store": self.store.state_dict(),
                "credit": {}, "gate": {}, "pending": []}


# ============================================================================ symbolic evaluation (no LLM)
def symbolic_predict(lib: SkillLibrary, matcher: SkillMatcher, res, cases: list, k_mod: int = 3) -> list[dict]:
    """Best firing formula skill (match x utility) -> core + fired modifications; then add the herbs of
    the k most reliable firing modification rules. No LLM; measures what the library itself knows."""
    formulas, mods = lib.active("formula"), lib.active("modification")
    out = []
    for c in cases:
        text, sym = res.text(c), set(c.symptoms)
        best = None
        for s in formulas:
            m = matcher.match(s, text, sym)
            if m >= matcher.fire and (best is None or m * s.utility > best[0]):
                best = (m * s.utility, s, m)
        herbs = matcher.execute(best[1], text, sym, best[2]) if best else []
        fired = sorted((s for s in mods if matcher.match(s, text, sym) >= matcher.fire), key=lambda s: -s.utility)[:k_mod]
        rule_herbs = dedup_keep_order([h for s in fired for h in s.add])
        out.append({"cid": c.cid, "formula": best[1].id if best else None, "formula_herbs": herbs,
                    "herbs": dedup_keep_order(herbs + [h for h in rule_herbs if h not in herbs]), "rule_herbs": rule_herbs})
    return out


def symbolic_report(res, cases: list, sym: list[dict], knn_herbs: list[list[str]]) -> str:
    cov = [i for i, s in enumerate(sym) if s["formula"]]
    f = lambda hs, idx: float(np.mean([prf(hs[i], cases[i].herbs)[2] for i in idx])) if idx else float("nan")
    allidx = list(range(len(cases)))
    rows = [
        ("方证技能（只用触发的那一条）", [s["formula_herbs"] for s in sym]),
        ("方证 + 加减规则（符号执行）", [s["herbs"] for s in sym]),
        ("kNN 基线（k=20，同一记忆）", knn_herbs),
    ]
    dedup = [i for i in allidx if not cases[i].flags.get("near_dup")]
    lines = [f"方证技能触发覆盖率：{len(cov)}/{len(cases)} = {len(cov) / max(len(cases), 1):.1%}",
             "| 预测 | F1 全部 | F1 去重 | F1 触发子集（n=%d） |" % len(cov), "|---|---|---|---|"]
    for name, hs in rows:
        lines.append(f"| {name} | {f(hs, allidx):.3f} | {f(hs, dedup):.3f} | {f(hs, cov):.3f} |")
    rule_p = [len(set(s["rule_herbs"]) & set(c.herbs)) / len(s["rule_herbs"]) for s, c in zip(sym, cases) if s["rule_herbs"]]
    lines.append(f"\n加减规则给出的药（每例前 3 条）：平均每例 {np.mean([len(s['rule_herbs']) for s in sym]):.1f} 味，命中原方的比例 {np.mean(rule_p) if rule_p else float('nan'):.3f}")
    return "\n".join(lines)
