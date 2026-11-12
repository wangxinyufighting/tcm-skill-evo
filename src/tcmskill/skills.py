"""Structured prescribing skills (理法方药), skill library, matching, symbolic execution, retrieval.

Branches
  formula       方证技能: triggers -> 病机 -> 治法 -> core herbs (君臣佐使) -> modifications
  modification  药症技能: trigger symptoms -> add herbs (cross-formula 随症加减 rule)
  principle     原则技能: free-text prescribing principle (not symbolically executable)
  free          自由文本技能 (SkeMex-style ablation; general/task/action in `free_branch`)
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

from .utils import dedup_keep_order, jaccard, truncate

ROLES = {"君", "臣", "佐", "使"}
BRANCH_PREFIX = {"formula": "F", "modification": "M", "principle": "P", "free": "S"}
_TERM_CLEAN = re.compile(r"[\s，,。.、；;：:\"'“”‘’（）()\[\]【】]+")


# ============================================================================ data classes
@dataclass
class Trigger:
    required: list[str] = field(default_factory=list)
    optional: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)

    def terms(self) -> list[str]:
        return dedup_keep_order(self.required + self.optional)

    def render(self) -> str:
        parts = []
        if self.required:
            parts.append("必见 " + "、".join(self.required))
        if self.optional:
            parts.append("或见 " + "、".join(self.optional))
        if self.exclude:
            parts.append("不宜 " + "、".join(self.exclude))
        return "；".join(parts)


@dataclass
class Skill:
    id: str = ""
    branch: str = "formula"
    name: str = ""
    triggers: list[Trigger] = field(default_factory=list)
    pathogenesis: str = ""
    principle: str = ""
    core: list[dict] = field(default_factory=list)  # [{"name": 药, "role": 君/臣/佐/使}]
    mods: list[dict] = field(default_factory=list)  # [{"if": [...], "add": [...], "remove": [...]}]
    add: list[str] = field(default_factory=list)  # modification branch
    context: str = ""
    content: str = ""  # principle / free-text body
    applies_to: str = ""  # principle: 适用场景 ; free: when
    free_branch: str = ""  # general / task / action (free-text skills)
    note: str = ""
    # ---- statistics / lifecycle
    utility: float = 0.5
    util_cat: dict = field(default_factory=dict)
    n_ret: int = 0
    n_used: int = 0
    n_credit: int = 0
    prec_ema: float = -1.0
    support: int = 0
    vstats: dict = field(default_factory=dict)
    status: str = "active"  # active | mature | deprecated
    created: int = 0
    updated: int = 0
    last_used: int = 0
    deprecated_at: int = -1
    version: int = 1
    evidence: list[str] = field(default_factory=list)
    usage_log: list[dict] = field(default_factory=list)
    history: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ helpers
    def core_herbs(self) -> list[str]:
        return [c["name"] for c in self.core]

    def all_terms(self) -> list[str]:
        out = []
        for t in self.triggers:
            out += t.terms()
        return dedup_keep_order(out)

    def herbs_mentioned(self) -> list[str]:
        hs = self.core_herbs() + list(self.add)
        for m in self.mods:
            hs += m.get("add", [])
        return dedup_keep_order(hs)

    def util_for(self, cat) -> float:
        return float(self.util_cat.get(str(cat), self.utility))

    def text_key(self) -> str:
        """Text used for similarity (free-text skills, principles, dedup)."""
        if self.branch == "free":
            return f"{self.name} {self.applies_to} {self.content}"
        if self.branch == "principle":
            return f"{self.applies_to} {self.content}"
        return f"{self.name} {' '.join(self.all_terms())} {self.pathogenesis} {self.principle} {' '.join(self.herbs_mentioned())}"

    def render(self, max_mods: int = 6) -> str:
        if self.branch == "formula":
            trig = " 或 ".join(f"（{t.render()}）" for t in self.triggers)
            core = " ".join(f"{c['name']}" + (f"({c['role']})" if c.get("role") else "") for c in self.core)
            mods = "；".join(
                f"若{'、'.join(m.get('if', []))}" + (f"加{'、'.join(m['add'])}" if m.get("add") else "") + (f"去{'、'.join(m['remove'])}" if m.get("remove") else "")
                for m in self.mods[:max_mods]
            )
            s = f"[{self.id}] 方证「{truncate(self.name, 20)}」｜适用：{trig}"
            if self.pathogenesis:
                s += f"｜病机：{truncate(self.pathogenesis, 30)}"
            if self.principle:
                s += f"｜治法：{truncate(self.principle, 20)}"
            s += f"｜基础方：{core}"
            if mods:
                s += f"｜加减：{mods}"
            if self.note:
                s += f"｜要点：{truncate(self.note, 40)}"
            return s
        if self.branch == "modification":
            trig = " 或 ".join(t.render() for t in self.triggers)
            s = f"[{self.id}] 随症加减｜{trig}" + (f"｜前提：{truncate(self.context, 30)}" if self.context else "") + f" → 加 {'、'.join(self.add)}"
            if self.vstats.get("n_xy"):
                s += f"（证据 {self.vstats['n_xy']} 例）"
            return s
        if self.branch == "principle":
            return f"[{self.id}] 开方原则｜{truncate(self.content, 120)}" + (f"（适用：{truncate(self.applies_to, 30)}）" if self.applies_to else "")
        return f"[{self.id}] {self.free_branch or 'skill'}「{truncate(self.name, 24)}」｜适用：{truncate(self.applies_to, 50)}｜{truncate(self.content, 220)}"

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Skill":
        d = dict(d)
        d["triggers"] = [Trigger(**t) for t in d.get("triggers", [])]
        return cls(**d)

    def mock_view(self) -> dict:
        return {"id": self.id, "branch": self.branch, "core": self.core_herbs(), "add": list(self.add), "mods": self.mods, "content": self.content}


# ============================================================================ library
class SkillLibrary:
    def __init__(self):
        self.skills: dict[str, Skill] = {}
        self.counters = {k: 0 for k in BRANCH_PREFIX}
        self.version = 0  # bumped on every change (used for caches)

    def __len__(self) -> int:
        return len(self.skills)

    def add(self, s: Skill, step: int = 0) -> Skill:
        self.counters[s.branch] += 1
        s.id = f"{BRANCH_PREFIX[s.branch]}{self.counters[s.branch]}"
        s.created = s.updated = s.last_used = step
        self.skills[s.id] = s
        self.version += 1
        return s

    def replace(self, s: Skill, step: int = 0, note: str = "") -> None:
        s.updated = step
        s.version += 1
        if note:
            s.history.append(f"{step}:{note}")
            s.history = s.history[-20:]
        self.skills[s.id] = s
        self.version += 1

    def get(self, sid: str) -> Skill | None:
        return self.skills.get(sid)

    def remove(self, sid: str) -> None:
        if sid in self.skills:
            del self.skills[sid]
            self.version += 1

    def active(self, branch: str | None = None) -> list[Skill]:
        return [s for s in self.skills.values() if s.status != "deprecated" and (branch is None or s.branch == branch)]

    def stats(self) -> dict:
        out: dict = {}
        for s in self.skills.values():
            b = out.setdefault(s.branch, {"active": 0, "mature": 0, "deprecated": 0})
            b[s.status] = b.get(s.status, 0) + 1
        return out

    def state_dict(self) -> dict:
        return {"counters": self.counters, "skills": [s.to_dict() for s in self.skills.values()]}

    def load_state_dict(self, d: dict) -> None:
        self.counters = {**{k: 0 for k in BRANCH_PREFIX}, **d.get("counters", {})}
        self.skills = {}
        for sd in d.get("skills", []):
            s = Skill.from_dict(sd)
            self.skills[s.id] = s
        self.version += 1


# ============================================================================ matching & execution
class SkillMatcher:
    """IDF-weighted trigger matching against what the agent sees (text + symptom keywords)."""

    def __init__(self, store, mcfg):
        self.store = store
        self.opt_w = float(mcfg.opt_weight)
        self.exc_pen = float(mcfg.exclude_penalty)
        self.fire = float(mcfg.fire)
        self.mod_if_ratio = float(mcfg.mod_if_ratio)
        self._idf_cache: dict[str, tuple[int, float]] = {}

    def w(self, term: str) -> float:
        hit = self._idf_cache.get(term)
        if hit is not None and hit[0] == self.store.n:
            return hit[1]
        v = self.store.idf(term)
        self._idf_cache[term] = (self.store.n, v)
        return v

    @staticmethod
    def present(term: str, text: str, symset) -> bool:
        return term in text or term in symset

    def match_trigger(self, t: Trigger, text: str, symset) -> float:
        if not t.required and not t.optional:
            return 0.0
        req_w = sum(self.w(x) for x in t.required)
        req_hit = sum(self.w(x) for x in t.required if self.present(x, text, symset))
        opt_w = sum(self.w(x) for x in t.optional)
        opt_hit = sum(self.w(x) for x in t.optional if self.present(x, text, symset))
        if t.required:
            req_cov = req_hit / req_w if req_w else 0.0
            if req_cov == 0:
                return 0.0
            score = req_cov if not t.optional else (1 - self.opt_w) * req_cov + self.opt_w * (opt_hit / opt_w if opt_w else 0.0)
        else:
            score = 0.7 * (opt_hit / opt_w if opt_w else 0.0)
        if any(self.present(x, text, symset) for x in t.exclude):
            score *= 1 - self.exc_pen
        return score

    def match(self, s: Skill, text: str, symset) -> float:
        if not s.triggers:
            return 0.0
        return max(self.match_trigger(t, text, symset) for t in s.triggers)

    def mod_fires(self, m: dict, text: str, symset) -> bool:
        cond = m.get("if", [])
        if not cond:
            return False
        hit = sum(1 for x in cond if self.present(x, text, symset))
        return hit / len(cond) >= self.mod_if_ratio - 1e-9

    def execute(self, s: Skill, text: str, symset, matched: float | None = None) -> list[str]:
        """Symbolic execution: herbs this skill would contribute to the case (no LLM)."""
        if s.branch not in ("formula", "modification"):
            return []
        m = self.match(s, text, symset) if matched is None else matched
        if m < self.fire:
            return []
        if s.branch == "modification":
            return list(s.add)
        herbs = s.core_herbs()
        removed: set[str] = set()
        for mod in s.mods:
            if self.mod_fires(mod, text, symset):
                herbs += mod.get("add", [])
                removed |= set(mod.get("remove", []))
        return [h for h in dedup_keep_order(herbs) if h not in removed]

    def matched_cases(self, s: Skill) -> list[int]:
        """Indices of seen cases on which the skill fires."""
        cands: set[int] = set()
        for t in s.triggers:
            for x in t.required or t.optional:
                cands |= self.store.term_post(x)
        st = self.store
        return [i for i in cands if self.match(s, st.texts[i], st.symsets[i]) >= self.fire]


# ============================================================================ retrieval
class SkillRetriever:
    def __init__(self, lib: SkillLibrary, matcher: SkillMatcher, res, scfg):
        self.lib = lib
        self.matcher = matcher
        self.res = res
        self.rc = scfg.retrieval
        self.structured = bool(scfg.structured)
        self._text_cache: dict = {"version": -1, "ids": [], "M": None}

    def _strength(self, s: Skill, step: int) -> float:
        return math.exp(-max(step - s.last_used, 0) / 3000.0)

    def _text_skills(self) -> tuple[list[Skill], object]:
        branches = ("principle",) if self.structured else ("free",)
        cache = self._text_cache
        if cache["version"] != self.lib.version:
            skills = [s for b in branches for s in self.lib.active(b)]
            M = self.res.vectorizer.transform([s.text_key() for s in skills]) if skills else None
            cache.update({"version": self.lib.version, "skills": skills, "M": M})
        return cache.get("skills", []), cache.get("M")

    def retrieve_batch(self, cases: list, texts: list[str], step: int = 0) -> list[list[tuple[Skill, float, float]]]:
        rc = self.rc
        text_skills, M = self._text_skills()
        sims = None
        if text_skills and M is not None:
            Q = self.res.vectorizer.transform(texts)
            sims = (Q @ M.T).toarray()
        formula = self.lib.active("formula") if self.structured else []
        mods = self.lib.active("modification") if self.structured else []
        out = []
        for ci, (case, text) in enumerate(zip(cases, texts)):
            symset = set(case.symptoms)
            picked: list[tuple[Skill, float, float]] = []
            for pool, k in ((formula, rc.k_formula), (mods, rc.k_modification)):
                scored = []
                for s in pool:
                    m = self.matcher.match(s, text, symset)
                    if m < rc.min_score:
                        continue
                    sc = rc.w_sim * m + rc.w_util * s.util_for(case.cat) + rc.w_strength * self._strength(s, step) + (rc.mature_bonus if s.status == "mature" else 0.0)
                    scored.append((s, sc, m))
                scored.sort(key=lambda x: -x[1])
                picked += scored[:k]
            if sims is not None:
                k = rc.k_principle if self.structured else rc.k_free
                scored = []
                for j, s in enumerate(text_skills):
                    m = float(sims[ci, j])
                    if m < float(rc.get("min_text_sim", 0.05)):
                        continue
                    sc = rc.w_sim * m + rc.w_util * s.util_for(case.cat) + rc.w_strength * self._strength(s, step) + (rc.mature_bonus if s.status == "mature" else 0.0)
                    scored.append((s, sc, m))
                scored.sort(key=lambda x: -x[1])
                picked += scored[:k]
            out.append(picked)
        return out


# ============================================================================ build / validate from LLM json
def clean_term(x) -> str:
    return _TERM_CLEAN.sub("", str(x or ""))


def _str_list(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        x = re.split(r"[、，,；;\s]+", x)
    if not isinstance(x, list):
        return []
    return [str(v) for v in x if str(v).strip()]


def ground_terms(terms: Iterable, grounding: list[str], min_len: int = 2, max_len: int = 8) -> list[str]:
    """Keep terms that literally appear in at least one grounding text."""
    out = []
    for t in terms:
        t = clean_term(t)
        if min_len <= len(t) <= max_len and (not grounding or any(t in g for g in grounding)):
            out.append(t)
    return dedup_keep_order(out)


def norm_herbs(names: Iterable, lexicon) -> list[str]:
    out = []
    for n in names:
        if isinstance(n, dict):
            n = n.get("name", "")
        out += lexicon.normalize(n)
    return dedup_keep_order(out)


def build_trigger(obj, grounding: list[str], min_len: int) -> Trigger | None:
    if isinstance(obj, str):
        obj = {"required": _str_list(obj)}
    if isinstance(obj, list):
        obj = {"required": obj}
    if not isinstance(obj, dict):
        return None
    t = Trigger(
        required=ground_terms(_str_list(obj.get("required")), grounding, min_len),
        optional=ground_terms(_str_list(obj.get("optional")), grounding, min_len),
        exclude=[clean_term(x) for x in _str_list(obj.get("exclude")) if len(clean_term(x)) >= min_len],
    )
    t.optional = [x for x in t.optional if x not in t.required]
    if not t.required and t.optional:
        t.required, t.optional = t.optional[:1], t.optional[1:]
    if not t.required:
        return None
    return t


def skill_from_json(obj: dict, lexicon, grounding: list[str], structured: bool = True, min_len: int = 2) -> tuple[Skill | None, str]:
    """Validate an LLM-proposed skill. Returns (skill or None, reason)."""
    if not isinstance(obj, dict):
        return None, "not_a_dict"
    br = str(obj.get("branch", "")).strip().lower()
    if not structured or br in ("general", "task", "action", "free"):
        title = truncate(str(obj.get("title") or obj.get("name") or ""), 40)
        content = truncate(str(obj.get("content") or ""), 400)
        if len(content) < 8:
            return None, "empty_content"
        return Skill(branch="free", free_branch=br if br in ("general", "task", "action") else "general", name=title, applies_to=truncate(str(obj.get("when") or obj.get("applies_to") or ""), 80), content=content), "ok"
    if br == "principle":
        content = truncate(str(obj.get("content") or ""), 200)
        if len(content) < 8:
            return None, "empty_content"
        return Skill(branch="principle", content=content, applies_to=truncate(str(obj.get("applies_to") or ""), 60)), "ok"
    trig = build_trigger(obj.get("trigger"), grounding, min_len)
    if trig is None:
        return None, "ungrounded_trigger"
    if br == "modification":
        add = norm_herbs(_str_list(obj.get("add")) if not isinstance(obj.get("add"), list) else obj.get("add"), lexicon)
        if not add:
            return None, "no_herbs"
        return Skill(branch="modification", triggers=[trig], add=add, context=truncate(str(obj.get("context") or ""), 60), note=truncate(str(obj.get("note") or ""), 80)), "ok"
    if br == "formula":
        core, seen = [], set()
        for c in obj.get("core_herbs") or []:
            name = c.get("name") if isinstance(c, dict) else c
            role = c.get("role", "") if isinstance(c, dict) else ""
            for lab in lexicon.normalize(name)[:1]:
                if lab not in seen:
                    seen.add(lab)
                    core.append({"name": lab, "role": role if role in ROLES else ""})
        if len(core) < 1:
            return None, "no_core_herbs"
        mods = []
        for m in obj.get("modifications") or []:
            if not isinstance(m, dict):
                continue
            cond = ground_terms(_str_list(m.get("if")), grounding, min_len)
            add = norm_herbs(m.get("add") or [], lexicon)
            rem = norm_herbs(m.get("remove") or [], lexicon)
            if cond and (add or rem):
                mods.append({"if": cond, "add": add, "remove": rem})
        return (
            Skill(
                branch="formula",
                name=truncate(str(obj.get("name") or ""), 30),
                triggers=[trig],
                pathogenesis=truncate(str(obj.get("pathogenesis") or ""), 60),
                principle=truncate(str(obj.get("principle") or ""), 40),
                core=core,
                mods=mods[:10],
                note=truncate(str(obj.get("note") or ""), 80),
            ),
            "ok",
        )
    return None, f"bad_branch:{br}"


def apply_patch(s: Skill, patch: dict, lexicon, grounding: list[str], min_len: int = 2) -> Skill:
    """Return a patched copy of a structured skill (fields that fail validation are ignored)."""
    new = copy.deepcopy(s)
    if not isinstance(patch, dict):
        return new
    if s.branch == "free":
        if patch.get("content"):
            new.content = truncate(str(patch["content"]), 400)
        if patch.get("when"):
            new.applies_to = truncate(str(patch["when"]), 80)
        return new
    if s.branch == "principle":
        if patch.get("content"):
            new.content = truncate(str(patch["content"]), 200)
        return new
    if patch.get("trigger"):
        t = build_trigger(patch["trigger"], grounding + [s.render()], min_len)
        if t is not None:
            new.triggers[0] = t
    if s.branch == "formula" and patch.get("core_herbs"):
        core = []
        for c in patch["core_herbs"]:
            name = c.get("name") if isinstance(c, dict) else c
            role = c.get("role", "") if isinstance(c, dict) else ""
            for lab in lexicon.normalize(name)[:1]:
                if lab not in [x["name"] for x in core]:
                    core.append({"name": lab, "role": role if role in ROLES else ""})
        if core:
            new.core = core
    if s.branch == "formula":
        for m in patch.get("add_modifications") or patch.get("modifications") or []:
            if not isinstance(m, dict):
                continue
            cond = ground_terms(_str_list(m.get("if")), grounding, min_len)
            add = norm_herbs(m.get("add") or [], lexicon)
            rem = norm_herbs(m.get("remove") or [], lexicon)
            if cond and (add or rem):
                new.mods.append({"if": cond, "add": add, "remove": rem})
        new.mods = merge_mods(new.mods)[:12]
    if s.branch == "modification" and patch.get("add"):
        add = norm_herbs(patch["add"], lexicon)
        if add:
            new.add = dedup_keep_order(new.add + add)
    if patch.get("note"):
        new.note = truncate(str(patch["note"]), 80)
    return new


def merge_mods(mods: list[dict]) -> list[dict]:
    by_cond: dict[tuple, dict] = {}
    for m in mods:
        key = tuple(sorted(m.get("if", [])))
        cur = by_cond.setdefault(key, {"if": list(m.get("if", [])), "add": [], "remove": []})
        cur["add"] = dedup_keep_order(cur["add"] + m.get("add", []))
        cur["remove"] = dedup_keep_order(cur["remove"] + m.get("remove", []))
    return list(by_cond.values())


def merge_skills(target: Skill, other: Skill, trig_jacc_thr: float = 0.4) -> Skill:
    """Programmatic merge (keeps target id). Different triggers + same formula = 异病同治 (multi-trigger)."""
    new = copy.deepcopy(target)
    if target.branch in ("formula", "modification"):
        for t in other.triggers:
            best = max((jaccard(t.terms(), u.terms()) for u in new.triggers), default=0.0)
            if best < trig_jacc_thr and len(new.triggers) < 4:
                new.triggers.append(copy.deepcopy(t))
            else:
                u = max(new.triggers, key=lambda u: jaccard(t.terms(), u.terms()))
                u.optional = [x for x in dedup_keep_order(u.optional + t.optional + [y for y in t.required if y not in u.required]) if x not in u.required][:8]
        if target.branch == "formula":
            new.mods = merge_mods(new.mods + other.mods)[:12]
        else:
            new.add = dedup_keep_order(new.add + other.add)
    elif other.content and other.content not in new.content and len(new.content) < 300:
        new.content = truncate(new.content + "；" + other.content, 400)
    new.evidence = dedup_keep_order(new.evidence + other.evidence)[-30:]
    new.n_used += other.n_used
    new.n_ret += other.n_ret
    new.utility = max(target.utility, other.utility) if target.n_credit + other.n_credit == 0 else (
        (target.utility * max(target.n_credit, 1) + other.utility * max(other.n_credit, 1)) / (max(target.n_credit, 1) + max(other.n_credit, 1))
    )
    new.n_credit += other.n_credit
    return new
