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
    meta: dict = field(default_factory=dict)  # style-specific details (select_edit: base / edits)

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
        base = {"structured": P.PRESCRIBE_STRUCTURED, "select_edit": P.PRESCRIBE_SELECT_EDIT}.get(self.style, P.PRESCRIBE_DIRECT)
        self.system = base + (P.vocab_block(labels) if pc.include_vocab else "")
        self.safety_mode = cfg.safety.mode
        self.select_final = str(pc.get("select_edit_final", "computed"))  # computed = 底方 − remove + add | herbs = 模型给的 herbs

    # ------------------------------------------------------------------ prompt
    def messages(self, case, ctx: dict) -> list[dict]:
        text = self.res.text(case)
        skills = "\n".join(s.render() for s in ctx.get("skills", []))
        if self.style == "select_edit":
            cands = "\n".join(
                P.render_candidate(i + 1, c["text"], c["herbs"], c.get("sim"), c.get("extra", "")) for i, c in enumerate(ctx.get("cases", []))
            )
            return [{"role": "system", "content": self.system}, {"role": "user", "content": P.select_edit_user(text, cands, skills)}]
        cases = "\n".join(
            P.render_case_line(i + 1, c["text"], c["herbs"], c.get("extra", "")) for i, c in enumerate(ctx.get("cases", []))
        )
        notes = ctx.get("notes", "")
        user = P.prescribe_user(text, skills=skills, cases=cases, notes=notes, notes_title=ctx.get("notes_title", "开方经验"))
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]

    # ------------------------------------------------------------------ parse
    def _labels(self, names, pred: Prediction) -> list[str]:
        if isinstance(names, str):
            names = re.split(r"[、，,；;\s]+", names)
        out = []
        for n in names if isinstance(names, list) else []:
            if isinstance(n, dict):
                n = n.get("name") or n.get("药名") or ""
            labs = self.res.lexicon.normalize(str(n))
            if not labs and str(n).strip():
                pred.oov.append(str(n).strip())
            out += labs
        return dedup_keep_order(out)

    def _parse_select_edit(self, case, data: dict, ctx: dict, pred: Prediction) -> Prediction:
        """选方加减: final = 底方 − remove + add (or the model's own `herbs` list, see prescribe.select_edit_final)."""
        cands = ctx.get("cases", [])
        m = re.search(r"[Cc]\s*(\d+)", str(data.get("base") or ""))
        bi = int(m.group(1)) - 1 if m else -1
        base = cands[bi] if 0 <= bi < len(cands) else None
        base_herbs = list(base["herbs"]) if base else []
        remove = set(self._labels(data.get("remove") or [], pred))
        add_raw = data.get("add") or []
        add_raw = add_raw if isinstance(add_raw, list) else [add_raw]
        add, cues = [], {}
        for it in add_raw:
            name = (it.get("name") or it.get("药名") or "") if isinstance(it, dict) else it
            for lab in self._labels([name], pred):
                if lab not in add:
                    add.append(lab)
                    if isinstance(it, dict) and it.get("cue"):
                        cues[lab] = truncate(str(it["cue"]), 12)
        llm_herbs = self._labels(data.get("herbs") or [], pred)
        if base is not None:
            computed = dedup_keep_order([h for h in base_herbs if h not in remove] + add)
        else:
            computed = llm_herbs or add
        final = llm_herbs if (self.select_final == "herbs" and llm_herbs) else (computed or llm_herbs)
        bset, aset = set(base_herbs), set(add)
        entries = []
        for h in final[: self.max_herbs]:
            src = "add" if h in aset and h not in bset else ("base" if h in bset else "self")
            entries.append({"name": h, "role": "", "source": src, "targets": [cues[h]] if h in cues else []})
        pred.meta = {
            "style": "select_edit", "base": f"C{bi + 1}" if base else "none", "base_cid": (base or {}).get("cid"),
            "base_sim": (base or {}).get("sim"), "n_remove": len(remove & bset), "n_add": len(aset - bset),
            "llm_herbs": llm_herbs, "consistent": set(llm_herbs) == set(computed), "fit": truncate(str(data.get("fit") or ""), 60),
        }
        return self._finish(pred, entries, data, set())

    def _finish(self, pred: Prediction, entries: list[dict], data: dict, valid: set) -> Prediction:
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

    def parse(self, case, data, ctx: dict) -> Prediction:
        pred = Prediction(cid=case.cid, retrieved=[s.id for s in ctx.get("skills", [])])
        if not isinstance(data, dict):
            pred.ok, pred.error = False, "no_json"
            return pred
        if self.style == "select_edit":
            return self._parse_select_edit(case, data, ctx, pred)
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
        return self._finish(pred, entries[: self.max_herbs], data, valid)

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
