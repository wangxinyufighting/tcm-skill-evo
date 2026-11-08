"""The prescribing agent: builds the prompt from context (skills / cases / notes), calls the
LLM, parses the structured prescription and normalises herb names into the label space."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from . import prompts as P
from .utils import dedup_keep_order, truncate

_SRC_RE = re.compile(r"[\[\]【】\s]")
_ID_RE = re.compile(r"[FMPS]\d+")


def _norm_src(x, valid: set) -> str:
    s = _SRC_RE.sub("", str(x or "self"))
    if s in valid or s in ("case", "note", "self"):
        return s
    m = _ID_RE.search(s)  # "技能F12" / "F12（麻黄汤）" -> F12
    if m and m.group(0) in valid:
        return m.group(0)
    if "医案" in s or "case" in s.lower():
        return "case"
    return "self"


@dataclass
class Prediction:
    cid: str
    herbs: list[str] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    used_skills: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    syndrome: str = ""
    principle: str = ""
    oov: list[str] = field(default_factory=list)
    unsafe: list[dict] = field(default_factory=list)
    removed_unsafe: list[dict] = field(default_factory=list)
    ok: bool = True
    error: str | None = None
    cached: bool = False
    skipped: bool = False
    json_status: str = "unknown"
    finish_reason: str | None = None
    truncated: bool = False
    empty_prescription: bool | None = None
    request_key: str = ""
    max_tokens: int = 0
    attempts: list[dict] = field(default_factory=list)
    rejected_herbs: list[str] = field(default_factory=list)
    request_failed: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Prediction":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in d.items() if k in keys})


class Prescriber:
    def __init__(self, llm, res, cfg):
        self.llm = llm
        self.res = res
        self.cfg = cfg
        pc = cfg.prescribe
        self.style = pc.style
        self.max_herbs = int(pc.max_herbs)
        labels = [h for h in res.labels if res.herb_freq.get(h, 0) >= int(pc.vocab_min_freq)]
        base = P.PRESCRIBE_STRUCTURED if self.style == "structured" else P.PRESCRIBE_DIRECT
        self.system = base + (P.vocab_block(labels) if pc.include_vocab else "")
        self.safety_mode = cfg.safety.mode

    # ------------------------------------------------------------------ prompt
    def messages(self, case, ctx: dict) -> list[dict]:
        text = self.res.text(case)
        skills = "\n".join(s.render() for s in ctx.get("skills", []))
        cases = "\n".join(
            P.render_case_line(i + 1, c["text"], c["herbs"], c.get("extra", "")) for i, c in enumerate(ctx.get("cases", []))
        )
        notes = ctx.get("notes", "")
        user = P.prescribe_user(text, skills=skills, cases=cases, notes=notes, notes_title=ctx.get("notes_title", "开方经验"))
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    # ------------------------------------------------------------------ parse
    def parse(self, case, data, ctx: dict) -> Prediction:
        pred = Prediction(cid=case.cid, retrieved=[s.id for s in ctx.get("skills", [])])
        if not isinstance(data, dict):
            pred.ok, pred.error = False, "no_json"
            return pred
        valid = set(pred.retrieved)
        raw = data.get("herbs") or data.get("处方") or data.get("prescription") or []
        if isinstance(raw, str):
            raw = re.split(r"[、，,；;\s]+", raw)
        if isinstance(raw, dict):  # {"君": ["麻黄"], "臣": [...]} -> flat list with roles
            flat = []
            for role, items in raw.items():
                for it in items if isinstance(items, list) else [items]:
                    flat.append({"name": it, "role": str(role)} if not isinstance(it, dict) else {**it, "role": it.get("role", role)})
            raw = flat
        entries, seen = [], set()
        for item in raw if isinstance(raw, list) else []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("药名") or item.get("herb") or ""
                role = str(item.get("role") or "")
                src = _norm_src(item.get("source"), valid)
                tg = item.get("targets") or []
            else:
                name, role, src, tg = str(item), "", "self", []
            labs = self.res.lexicon.normalize(name)
            if not labs:
                if str(name).strip():
                    pred.oov.append(str(name).strip())
                continue
            for lab in labs:
                if lab not in seen:
                    seen.add(lab)
                    entries.append({"name": lab, "role": role[:2], "source": src, "targets": [truncate(str(t), 12) for t in tg][:5] if isinstance(tg, list) else []})
        entries = entries[: self.max_herbs]
        herbs = [e["name"] for e in entries]
        pred.unsafe = self.res.incompat.violations(herbs)
        if self.safety_mode == "filter" and pred.unsafe:
            kept, removed = self.res.incompat.filter(herbs)
            keep = set(kept)
            entries = [e for e in entries if e["name"] in keep]
            herbs = [e["name"] for e in entries]
            pred.removed_unsafe = removed
        used = [_norm_src(x, valid) for x in (data.get("used_skills") or []) if isinstance(x, (str, int))]
        used = [x for x in used if x in valid]
        used = dedup_keep_order(used + [e["source"] for e in entries if e["source"] in valid])
        pred.entries, pred.herbs, pred.used_skills = entries, herbs, used
        pred.syndrome = truncate(str(data.get("syndrome") or ""), 80)
        pred.principle = truncate(str(data.get("principle") or ""), 60)
        pred.ok = bool(herbs)
        if not herbs:
            pred.error = "empty_prescription"
        return pred

    # ------------------------------------------------------------------ run
    def prescribe(self, case, ctx: dict) -> Prediction:
        msgs = self.messages(case, ctx)
        meta = {
            "cid": case.cid,
            "text": self.res.text(case),
            "symptoms": case.symptoms,
            "skills": [s.mock_view() for s in ctx.get("skills", [])],
            "cases": [{"herbs": c["herbs"]} for c in ctx.get("cases", [])],
            "fallback": self.res.labels[: max(self.res.median_len, 3)],
            "style": self.style,
        }
        r = self.llm.chat_json(msgs, task="prescribe", meta=meta)
        pred = self.parse(case, r.data, ctx)
        pred.cached = r.cached
        pred.request_failed = r.request_failed
        pred.json_status, pred.finish_reason = r.json_status, r.finish_reason
        pred.truncated, pred.max_tokens = r.truncated, r.max_tokens
        pred.request_key, pred.attempts = r.request_key, r.attempts
        pred.empty_prescription = not bool(pred.herbs) if isinstance(r.data, dict) else None
        if not r.ok:
            pred.ok, pred.error = False, r.error or pred.error
            pred.rejected_herbs = pred.herbs
            pred.herbs, pred.entries, pred.used_skills = [], [], []
            pred.unsafe, pred.removed_unsafe = [], []
        return pred

    def prescribe_batch(self, cases: list, ctxs: list[dict], desc: str = "prescribe") -> list[Prediction]:
        return self.llm.map(lambda i: self.prescribe(cases[i], ctxs[i]), list(range(len(cases))), desc=desc)
