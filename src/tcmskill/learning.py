"""Experience records, herb-level credit assignment / utility updates, reflection, buffer gating."""
from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

from . import prompts as P
from .metrics import prf


# ============================================================================ records
@dataclass
class ExperienceRecord:
    cid: str
    cat: int
    text: str
    symptoms: list[str]
    gt: list[str]
    pred: list[str] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    used: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    syndrome: str = ""
    principle: str = ""
    f1: float = -1.0
    tp: list[str] = field(default_factory=list)
    fn: list[str] = field(default_factory=list)
    fp: list[str] = field(default_factory=list)
    attempted: bool = True
    ok: bool = True
    step: int = 0
    window: int = 0
    reflection: dict | None = None
    priority: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperienceRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_record(case, text: str, pred, step: int, window: int) -> ExperienceRecord:
    gt = list(case.herbs)
    rec = ExperienceRecord(cid=case.cid, cat=case.cat, text=text, symptoms=list(case.symptoms), gt=gt, step=step, window=window)
    if pred is None:
        rec.attempted = False
        return rec
    P_ = pred.herbs
    rec.pred, rec.entries, rec.used, rec.retrieved = list(P_), list(pred.entries), list(pred.used_skills), list(pred.retrieved)
    rec.syndrome, rec.principle = pred.syndrome, pred.principle
    rec.ok = bool(pred.ok)
    G = set(gt)
    rec.tp = [h for h in P_ if h in G]
    rec.fp = [h for h in P_ if h not in G]
    rec.fn = [h for h in gt if h not in set(P_)]
    rec.f1 = prf(P_, gt)[2]
    return rec


# ============================================================================ credit
class CreditAssigner:
    """Herb-level (default) or episode-level credit; utilities updated once per window."""

    def __init__(self, scfg, matcher):
        self.mode = scfg.credit
        self.u = scfg.utility
        self.matcher = matcher
        self.base_prec: dict[str, float] = {}
        self.base_f1: dict[str, float] = {}
        self.pending: dict[str, list[tuple[float, int]]] = defaultdict(list)

    def _b(self, d: dict, key: str, default: float = 0.3) -> float:
        return d.get(key, default)

    def _upd(self, d: dict, key: str, v: float) -> None:
        a = float(self.u.baseline_ema)
        d[key] = (1 - a) * d.get(key, v) + a * v

    def _signed(self, a: float) -> float:
        return a if a >= 0 else (1 + float(self.u.harm)) * a

    def observe(self, case, text: str, rec: ExperienceRecord, retrieved: list, step: int) -> None:
        if not rec.attempted or not rec.ok:
            return
        G, P_ = set(rec.gt), set(rec.pred)
        cat = case.cat
        used = set(rec.used)
        symset = set(case.symptoms)
        for s in retrieved:
            s.n_ret += 1
            if s.id in used:
                C = [e["name"] for e in rec.entries if e.get("source") == s.id]
                if self.mode == "herb" and C and s.branch in ("formula", "modification", "free"):
                    prec = len(set(C) & G) / len(C)
                    key = f"{cat}|{s.branch}"
                    a = prec - self._b(self.base_prec, key)
                    c = self._signed(a)
                    self._upd(self.base_prec, key, prec)
                    s.prec_ema = prec if s.prec_ema < 0 else 0.8 * s.prec_ema + 0.2 * prec
                    val = prec
                else:
                    a = rec.f1 - self._b(self.base_f1, str(cat))
                    c = self._signed(a) * (float(self.u.principle_scale) if s.branch == "principle" else 1.0)
                    val = rec.f1
                self.pending[s.id].append((c, cat))
                s.n_used += 1
                s.last_used = step
                s.evidence = (s.evidence + [case.cid])[-30:]
                s.usage_log = (s.usage_log + [{"cid": case.cid, "step": step, "adopted": True, "prec": round(val, 3), "cat": cat}])[-60:]
            elif self.mode == "herb" and s.branch in ("formula", "modification"):
                E = self.matcher.execute(s, text, symset)
                if E:
                    prec_e = len(set(E) & G) / len(E)
                    gain = len(set(E) & (G - P_))
                    a = prec_e - self._b(self.base_prec, f"{cat}|{s.branch}")
                    if gain >= 1 and a >= float(self.u.miss_delta):
                        self.pending[s.id].append((float(self.u.miss_gain) * a, cat))
                        s.usage_log = (s.usage_log + [{"cid": case.cid, "step": step, "adopted": False, "prec": round(prec_e, 3), "cat": cat}])[-60:]
        self._upd(self.base_f1, str(cat), rec.f1)

    def apply(self, lib) -> int:
        n = 0
        for sid, lst in self.pending.items():
            s = lib.get(sid)
            if s is None or not lst:
                continue
            k = min(s.n_credit, int(self.u.warmup))
            eta = float(self.u.eta_min) + (float(self.u.eta_max) - float(self.u.eta_min)) * (1 - math.cos(math.pi * k / max(int(self.u.warmup), 1))) / 2
            before = s.utility
            s.utility = min(1.0, max(0.0, s.utility + eta * sum(c for c, _ in lst) / len(lst)))
            by_cat: dict[int, list[float]] = defaultdict(list)
            for c, cat in lst:
                by_cat[cat].append(c)
            for cat, cs in by_cat.items():
                cur = s.util_cat.get(str(cat), before)
                s.util_cat[str(cat)] = min(1.0, max(0.0, cur + eta * sum(cs) / len(cs)))
            s.n_credit += len(lst)
            n += 1
        self.pending.clear()
        return n

    def state_dict(self) -> dict:
        return {"base_prec": self.base_prec, "base_f1": self.base_f1}

    def load_state_dict(self, d: dict) -> None:
        self.base_prec = dict(d.get("base_prec", {}))
        self.base_f1 = dict(d.get("base_f1", {}))


# ============================================================================ gating
class Gate:
    """Select informative records for (costly) reflection/induction, balancing categories."""

    def __init__(self, gcfg, seed: int = 0):
        self.g = gcfg
        self.rng = random.Random(seed)

    def state_dict(self) -> dict:
        return {"rng": self.rng.getstate()}

    def load_state_dict(self, d: dict) -> None:
        if "rng" in d:
            version, state, gaussian = d["rng"]
            self.rng.setstate((version, tuple(state), gaussian))

    def select(self, records: list[ExperienceRecord]) -> list[ExperienceRecord]:
        cand = [r for r in records if r.ok and r.gt]
        if not cand:
            return []
        attempted = [r for r in cand if r.attempted]
        if attempted:
            fails = [r for r in attempted if r.f1 < float(self.g.success_f1)]
            succ = [r for r in attempted if r.f1 >= float(self.g.success_f1)]
            k_s = int(math.ceil(len(succ) * float(self.g.success_keep)))
            pool = fails + self.rng.sample(succ, min(k_s, len(succ)))
            for r in pool:
                r.priority = float(self.g.alpha) * math.log(1 + len(r.fn) + len(r.fp)) + float(self.g.beta) * (1.0 if r.retrieved else 0.0)
        else:  # hindsight-only: no attempt, prefer richer prescriptions
            pool = list(cand)
            for r in pool:
                r.priority = math.log(1 + len(r.gt)) + 0.01 * self.rng.random()
        chosen: list[ExperienceRecord] = []
        cat_n: Counter = Counter()
        remaining = sorted(pool, key=lambda r: -r.priority)
        while remaining and len(chosen) < int(self.g.max_per_window):
            best = max(remaining, key=lambda r: r.priority / (1 + cat_n[r.cat]))
            remaining.remove(best)
            chosen.append(best)
            cat_n[best.cat] += 1
        return chosen


# ============================================================================ reflection
class Reflector:
    def __init__(self, llm, res):
        self.llm = llm
        self.res = res

    def reflect(self, rec: ExperienceRecord, skills_text: str = "") -> dict | None:
        hindsight = not rec.attempted
        user = P.reflect_user(
            rec.text,
            rec.gt,
            pred=None if hindsight else rec.pred,
            syndrome=rec.syndrome,
            principle=rec.principle,
            pred_entries=rec.entries,
            tp=rec.tp,
            fn=rec.fn,
            fp=rec.fp,
            skills=skills_text,
        )
        meta = {"cid": rec.cid, "text": rec.text, "symptoms": rec.symptoms, "gt": rec.gt, "fn": rec.fn, "fp": rec.fp, "hindsight": hindsight}
        r = self.llm.chat_json([{"role": "system", "content": P.REFLECT_SYSTEM}, {"role": "user", "content": user}], task="reflect", meta=meta)
        return r.data if r.ok and isinstance(r.data, dict) else None

    def reflect_batch(self, recs: list[ExperienceRecord], lib=None) -> None:
        def run(rec):
            skills_text = ""
            if lib is not None and rec.retrieved:
                skills_text = "\n".join(lib.get(s).render() for s in rec.retrieved if lib.get(s) is not None)
            rec.reflection = self.reflect(rec, skills_text)

        self.llm.map(run, recs, desc="reflect")


def reflection_cues(rec: ExperienceRecord, lexicon) -> list[tuple[str, list[str], str]]:
    """(herb, cue terms, reason) proposals from a reflection: missed herbs (attempted) or GT modules (hindsight)."""
    out = []
    ref = rec.reflection or {}
    for m in ref.get("missed", []) or []:
        if not isinstance(m, dict):
            continue
        labs = lexicon.normalize(m.get("herb", ""))
        cues = m.get("cue") or []
        if isinstance(cues, str):
            cues = [cues]
        if labs and cues:
            out.append((labs[0], [str(c) for c in cues], str(m.get("reason", ""))[:60]))
    if not rec.attempted:
        for mod in (ref.get("gt_analysis") or {}).get("modules", []) or []:
            if not isinstance(mod, dict):
                continue
            cues = mod.get("targets") or []
            if isinstance(cues, str):
                cues = [cues]
            for h in mod.get("herbs", []) or []:
                labs = lexicon.normalize(h)
                if labs and cues:
                    out.append((labs[0], [str(c) for c in cues], str(mod.get("function", ""))[:60]))
    return out
