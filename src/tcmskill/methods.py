"""All LLM methods behind one interface (same backbone, same prompt format, same evaluation).

    predict(cases, frozen)  -> list[Prediction]   (uses the current memory snapshot)
    observe(cases, preds)   -> dict               (learning step after GT is revealed)
    state_dict / load_state_dict                  (checkpointing)
"""
from __future__ import annotations

import logging
from collections import Counter

from . import prompts as P
from .govern import Governor
from .induce import Clusterer, Inducer, Verifier, Writer, rules_from_reflections
from .learning import CreditAssigner, ExperienceRecord, Gate, Reflector, make_record
from .prescriber import Prescriber
from .skills import SkillLibrary, SkillMatcher, SkillRetriever
from .stats import SeenStore
from .utils import truncate

LOG = logging.getLogger(__name__)


class BaseMethod:
    learns = False
    needs_attempt = True

    def __init__(self, cfg, llm, res):
        self.cfg, self.llm, self.res = cfg, llm, res
        self.prescriber = Prescriber(llm, res, cfg)
        self.step = 0
        self.window = 0

    def warmup(self, train_ids: list[str]) -> None:
        """Optional offline preparation (e.g. case_rag with memory_scope=all_train)."""

    def contexts(self, cases: list, frozen: bool) -> list[dict]:
        return [{} for _ in cases]

    def predict(self, cases: list, frozen: bool = False):
        return self.prescriber.prescribe_batch(cases, self.contexts(cases, frozen))

    def observe(self, cases: list, preds: list) -> dict:
        self.step += len(cases)
        self.window += 1
        return {}

    def summary(self) -> dict:
        return {}

    def state_dict(self) -> dict:
        return {"step": self.step, "window": self.window}

    def load_state_dict(self, d: dict) -> None:
        self.step, self.window = d.get("step", 0), d.get("window", 0)

    # ------------------------------------------------------------------ shared helpers
    def _case_ctx(self, store: SeenStore, cases: list, k: int, extra=None) -> list[list[dict]]:
        if store.n == 0 or k <= 0:
            return [[] for _ in cases]
        texts = [self.res.text(c) for c in cases]
        nbrs = store.neighbors_by_text(texts, k, exclude=[{c.cid} for c in cases])
        out = []
        for lst in nbrs:
            out.append([{"cid": cid, "sim": round(float(sim), 3), "text": store.texts[store.pos[cid]], "herbs": self.res.cases[cid].herbs,
                         "extra": extra(cid) if extra else ""} for cid, sim in lst])
        return out


# ============================================================================ static baselines
class ZeroShot(BaseMethod):
    pass


class CaseRAG(BaseMethod):
    learns = True

    def __init__(self, cfg, llm, res):
        super().__init__(cfg, llm, res)
        self.store = SeenStore(res)
        self.k = int(cfg.prescribe.n_cases)

    def warmup(self, train_ids):
        if self.cfg.get_path("case_rag.memory_scope", "seen") == "all_train":
            self.store.add(self.res.get(self.res.splits["train"]))  # the whole training split (no val/test)

    def contexts(self, cases, frozen):
        return [{"cases": c} for c in self._case_ctx(self.store, cases, self.k)]

    def observe(self, cases, preds):
        self.store.add(cases)
        return super().observe(cases, preds)

    def summary(self):
        return {"memory_cases": self.store.n}

    def state_dict(self):
        return {**super().state_dict(), "store": self.store.state_dict()}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.store = SeenStore(self.res)
        self.store.load_state_dict(d.get("store", {}))


# ============================================================================ memory baselines
class ReflexionLite(BaseMethod):
    """Store every observed case; reflect on failures; retrieve similar cases with their lessons."""

    learns = True

    def __init__(self, cfg, llm, res):
        super().__init__(cfg, llm, res)
        self.store = SeenStore(res)
        self.lessons: dict[str, str] = {}
        self.k = int(cfg.baselines_mem.reflexion_k)
        self.gate = Gate(cfg.skills.gate, seed=cfg.data.seed)
        self.reflector = Reflector(llm, res)

    def contexts(self, cases, frozen):
        ex = lambda cid: f"经验：{self.lessons[cid]}" if cid in self.lessons else ""
        return [{"cases": c} for c in self._case_ctx(self.store, cases, self.k, ex)]

    def observe(self, cases, preds):
        recs = [make_record(c, self.res.text(c), p, self.step + i, self.window) for i, (c, p) in enumerate(zip(cases, preds))]
        kept = self.gate.select(recs)
        self.reflector.reflect_batch(kept)
        for r in kept:
            lesson = (r.reflection or {}).get("lesson")
            if lesson:
                self.lessons[r.cid] = truncate(str(lesson), 80)
        self.store.add(cases)
        return {**super().observe(cases, preds), "n_reflected": len(kept)}

    def summary(self):
        return {"memory_cases": self.store.n, "lessons": len(self.lessons)}

    def state_dict(self):
        return {**super().state_dict(), "store": self.store.state_dict(), "lessons": self.lessons,
                "gate": self.gate.state_dict()}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.store = SeenStore(self.res)
        self.store.load_state_dict(d.get("store", {}))
        self.lessons = dict(d.get("lessons", {}))
        self.gate.load_state_dict(d.get("gate", {}))


class ExpeLLite(BaseMethod):
    """Global insight list (ADD/EDIT/AGREE/REMOVE with votes) + retrieval of successful cases."""

    learns = True

    def __init__(self, cfg, llm, res):
        super().__init__(cfg, llm, res)
        self.insights: list[dict] = []
        self.success = SeenStore(res)
        self.max_ins = int(cfg.baselines_mem.expel_max_insights)
        self.k = int(cfg.baselines_mem.expel_k_success)

    def _notes(self) -> str:
        top = sorted(self.insights, key=lambda x: -x["votes"])[: self.max_ins]
        return "\n".join(f"{i + 1}. {x['text']}" for i, x in enumerate(top))

    def contexts(self, cases, frozen):
        notes = self._notes()
        cs = self._case_ctx(self.success, cases, self.k)
        return [{"notes": notes, "cases": c} for c in cs]

    def observe(self, cases, preds):
        recs = [make_record(c, self.res.text(c), p, self.step + i, self.window) for i, (c, p) in enumerate(zip(cases, preds))]
        fails = sorted([r for r in recs if r.ok and r.f1 < 0.8], key=lambda r: r.f1)[:10]
        if fails:
            items = [{"text": r.text, "pred": r.pred, "gt": r.gt, "fn": r.fn, "fp": r.fp} for r in fails]
            ordered = sorted(self.insights, key=lambda x: -x["votes"])[: self.max_ins]
            listing = "\n".join(f"{i + 1}. {x['text']}（票数 {x['votes']}）" for i, x in enumerate(ordered)) or "（空）"
            user = f"【当前经验清单】\n{listing}\n\n【本轮开方结果】\n{P.comparison_lines(items)}\n\n请输出 JSON。"
            r = self.llm.chat_json([{"role": "system", "content": P.EXPEL_SYSTEM}, {"role": "user", "content": user}], task="expel", meta={"cids": [r.cid for r in fails], "items": items, "n_insights": len(ordered)})
            if r.ok and isinstance(r.data, dict):
                self._apply(r.data.get("operations") or [], ordered)
        self.success.add([c for c, rr in zip(cases, recs) if rr.ok and rr.f1 >= 0.5])
        return {**super().observe(cases, preds), "insights": len(self.insights)}

    def _apply(self, ops, ordered):
        for op in ops:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "")).upper()
            try:
                idx = int(op.get("id", 0)) - 1
            except (TypeError, ValueError):
                idx = -1
            tgt = ordered[idx] if 0 <= idx < len(ordered) else None
            if kind == "ADD" and op.get("text"):
                self.insights.append({"text": truncate(str(op["text"]), 100), "votes": 2})
            elif kind == "AGREE" and tgt:
                tgt["votes"] += 1
            elif kind == "EDIT" and tgt and op.get("text"):
                tgt["text"] = truncate(str(op["text"]), 100)
            elif kind == "REMOVE" and tgt:
                tgt["votes"] -= 2
        self.insights = [x for x in self.insights if x["votes"] > 0]
        self.insights = sorted(self.insights, key=lambda x: -x["votes"])[: 2 * self.max_ins]

    def summary(self):
        return {"insights": len(self.insights), "success_cases": self.success.n}

    def state_dict(self):
        return {**super().state_dict(), "insights": self.insights, "success": self.success.state_dict()}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.insights = list(d.get("insights", []))
        self.success = SeenStore(self.res)
        self.success.load_state_dict(d.get("success", {}))


class DCLite(BaseMethod):
    """Dynamic-Cheatsheet style: one cheatsheet rewritten after every window."""

    learns = True

    def __init__(self, cfg, llm, res):
        super().__init__(cfg, llm, res)
        self.cheatsheet = ""
        self.max_chars = int(cfg.baselines_mem.dc_max_chars)

    def contexts(self, cases, frozen):
        return [{"notes": self.cheatsheet, "notes_title": "开方备忘录"} for _ in cases]

    def observe(self, cases, preds):
        recs = [make_record(c, self.res.text(c), p, self.step + i, self.window) for i, (c, p) in enumerate(zip(cases, preds))]
        fails = sorted([r for r in recs if r.ok and r.f1 < 0.8], key=lambda r: r.f1)[:12]
        if fails:
            items = [{"text": r.text, "pred": r.pred, "gt": r.gt, "fn": r.fn, "fp": r.fp} for r in fails]
            user = f"【当前备忘录】\n{self.cheatsheet or '（空）'}\n\n【本轮开方结果】\n{P.comparison_lines(items)}\n\n请输出 JSON。"
            sysmsg = P.DC_SYSTEM.format(max_chars=self.max_chars)
            r = self.llm.chat_json([{"role": "system", "content": sysmsg}, {"role": "user", "content": user}], task="dc", meta={"cids": [r.cid for r in fails], "cheatsheet": self.cheatsheet, "items": items})
            if r.ok and isinstance(r.data, dict) and r.data.get("cheatsheet"):
                self.cheatsheet = str(r.data["cheatsheet"])[: self.max_chars]
        return {**super().observe(cases, preds), "cheatsheet_chars": len(self.cheatsheet)}

    def summary(self):
        return {"cheatsheet_chars": len(self.cheatsheet)}

    def state_dict(self):
        return {**super().state_dict(), "cheatsheet": self.cheatsheet}

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.cheatsheet = d.get("cheatsheet", "")


# ============================================================================ ours
class SkillMethod(BaseMethod):
    learns = True

    def __init__(self, cfg, llm, res):
        super().__init__(cfg, llm, res)
        sc = cfg.skills
        self.sc = sc
        self.needs_attempt = bool(sc.attempt_first)
        self.lib = SkillLibrary()
        self.store = SeenStore(res)
        self.matcher = SkillMatcher(self.store, sc.match)
        self.retriever = SkillRetriever(self.lib, self.matcher, res, sc)
        self.credit = CreditAssigner(sc, self.matcher)
        self.gate = Gate(sc.gate, seed=cfg.data.seed)
        self.reflector = Reflector(llm, res)
        labels = [h for h in res.labels if res.herb_freq.get(h, 0) >= int(cfg.prescribe.vocab_min_freq)]
        vocab_text = P.vocab_block(labels)
        self.inducer = Inducer(llm, res, self.store, self.lib, self.matcher, sc, vocab_text)
        self.verifier = Verifier(self.store, self.matcher, res, sc, cfg.safety.mode)
        self.writer = Writer(self.lib, self.verifier, res, sc)
        self.clusterer = Clusterer(sc.cluster, single=sc.induction == "single")
        self.governor = Governor(self.lib, self.store, self.matcher, self.verifier, llm, res, sc, vocab_text)
        self.pending: list[ExperienceRecord] = []
        self._last_ctx: dict[str, list] = {}
        self._link_cache: tuple = (-1, {})

    def _links(self) -> dict[str, list[str]]:
        """case id -> formula skills induced from it (skill.evidence); used by skills.link_cases."""
        if self._link_cache[0] != self.lib.version:
            m: dict[str, list[str]] = {}
            for s in self.lib.active("formula"):
                for cid in s.evidence:
                    m.setdefault(cid, []).append(s.id)
            self._link_cache = (self.lib.version, m)
        return self._link_cache[1]

    def contexts(self, cases, frozen):
        texts = [self.res.text(c) for c in cases]
        picked = self.retriever.retrieve_batch(cases, texts, self.step)
        ctxs = [{"skills": [s for s, _, _ in p]} for p in picked]
        k = int(self.sc.hybrid_cases)
        if k > 0:
            link = bool(self.sc.get("link_cases", False))
            links = self._links() if link else {}
            extra = (lambda cid: ("属方族：" + "、".join(links[cid])) if cid in links else "") if link else None
            for ctx, cs in zip(ctxs, self._case_ctx(self.store, cases, k, extra)):
                ctx["cases"] = cs
                if link:  # the families of the shown cases first, then trigger-matched skills
                    have = {s.id for s in ctx["skills"]}
                    linked = []
                    for c in cs:
                        for sid in links.get(c["cid"], []):
                            if sid not in have and self.lib.get(sid) is not None:
                                have.add(sid)
                                linked.append(self.lib.get(sid))
                    ctx["skills"] = linked[: int(self.sc.get("link_max", 5))] + ctx["skills"]
        if not frozen:
            self._last_ctx = {c.cid: ctx["skills"] for c, ctx in zip(cases, ctxs)}
        return ctxs

    def observe(self, cases, preds):
        sc = self.sc
        step_end = self.step + len(cases)
        recs = []
        for i, (c, p) in enumerate(zip(cases, preds)):
            text = self.res.text(c)
            rec = make_record(c, text, p if self.needs_attempt else None, self.step + i, self.window)
            if rec.attempted:
                self.credit.observe(c, text, rec, self._last_ctx.get(c.cid, []), self.step + i)
            recs.append(rec)
        n_credited = self.credit.apply(self.lib)
        self.store.add(cases)  # GT revealed -> statistics can use these cases from now on

        kept = self.gate.select(recs)
        self.reflector.reflect_batch(kept, self.lib)
        stats: Counter = Counter()
        if sc.rules_from_reflection and sc.structured:
            stats.update(self.writer.apply(rules_from_reflections(kept, self.res.lexicon, int(sc.verify_thr.min_term_len)), [], [], step_end))
        keep_ids = {r.cid for r in kept}
        pend = [r for r in self.pending if self.window - r.window <= int(sc.cluster.pending_windows) and r.cid not in keep_ids]
        clusters, leftovers = self.clusterer.build(kept + pend, self.store)
        for out in self.inducer.induce_batch(clusters):
            stats.update(self.writer.apply(out["ops"], out["grounding"], out["cids"], step_end))
        pend_new = [r for r in leftovers if (not r.attempted) or r.f1 < float(sc.gate.success_f1)]
        cap = 3 * int(sc.gate.max_per_window)
        self.pending = sorted(pend_new, key=lambda r: (-r.window, -r.priority))[:cap]
        gov = {}
        if int(sc.govern_every) > 0 and (self.window + 1) % int(sc.govern_every) == 0:
            gov = self.governor.run(self.window, step_end)
        self.step = step_end
        self.window += 1
        return {"kept": len(kept), "clusters": len(clusters), "credited": n_credited, "write": dict(stats), "govern": gov}

    def summary(self):
        return {"skills": self.lib.stats(), "n_active": len(self.lib.active()), "seen": self.store.n, "pending": len(self.pending)}

    def state_dict(self):
        return {
            **super().state_dict(),
            "library": self.lib.state_dict(),
            "store": self.store.state_dict(),
            "credit": self.credit.state_dict(),
            "gate": self.gate.state_dict(),
            "pending": [r.to_dict() for r in self.pending],
        }

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.lib.load_state_dict(d.get("library", {}))
        self.store.load_state_dict(d.get("store", {}))
        self.credit.load_state_dict(d.get("credit", {}))
        self.gate.load_state_dict(d.get("gate", {}))
        self.pending = [ExperienceRecord.from_dict(x) for x in d.get("pending", [])]


METHODS = {
    "zeroshot": ZeroShot,
    "case_rag": CaseRAG,
    "reflexion": ReflexionLite,
    "expel": ExpeLLite,
    "dc": DCLite,
    "skills": SkillMethod,
}


def build_method(cfg, llm, res) -> BaseMethod:
    t = cfg.method.type
    if t not in METHODS:
        raise KeyError(f"unknown method type {t}")
    return METHODS[t](cfg, llm, res)
