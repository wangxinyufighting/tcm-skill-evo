"""Deterministic mock LLM for smoke tests and cost estimation (`--llm mock`).

It never sees the prompt semantics; callers pass structured `meta`, and the mock produces
plausible outputs from it (e.g. it prescribes the core herbs of retrieved skills). Token usage
is estimated from the *real* prompt text, so `--llm mock` gives a realistic cost estimate.
"""
from __future__ import annotations

import json
import re
import threading

from .llm import BaseLLM, LLMResult
from .utils import estimate_tokens

TYPICAL_OUT = {"prescribe": 380, "reflect": 480, "induce": 950, "split": 700, "expel": 500, "dc": 1500}


class MockLLM(BaseLLM):
    name = "mock"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._seen = set()
        self._lock = threading.Lock()

    def chat_json(self, messages, task, meta=None, max_tokens=None) -> LLMResult:
        meta = meta or {}
        handler = getattr(self, f"_h_{task}", None)
        data = handler(meta) if handler else {}
        text = json.dumps(data, ensure_ascii=False)
        sys_txt = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
        rest = "".join(m["content"] for m in messages if m is not messages[0]) if sys_txt else "".join(m["content"] for m in messages)
        s_tok, r_tok = estimate_tokens(sys_txt), estimate_tokens(rest)
        key = (task, hash(sys_txt))
        with self._lock:
            hit = key in self._seen
            self._seen.add(key)
        usage = {
            "prompt_tokens": s_tok + r_tok,
            "prompt_cache_hit_tokens": s_tok if hit else 0,
            "prompt_cache_miss_tokens": r_tok + (0 if hit else s_tok),
            "completion_tokens": TYPICAL_OUT.get(task, 400),
        }
        budget = self.output_budget(max_tokens)
        res = LLMResult(text=text, data=data, ok=True, model="mock", usage=usage,
                        finish_reason="stop", json_status="complete", max_tokens=budget,
                        attempts=[{"attempt": 1, "ok": True, "model": "mock", "finish_reason": "stop",
                                   "json_status": "complete", "truncated": False, "max_tokens": budget,
                                   "usage": usage, "error": None, "text_chars": len(text)}])
        self.usage.add(task, usage, cached=False, error=False, model=self.model)
        self._log_call(task, res, meta)
        return res

    # ------------------------------------------------------------------ handlers
    @staticmethod
    def _h_prescribe(m: dict) -> dict:
        text = m.get("text", "")
        herbs: list[tuple[str, str]] = []
        seen = set()

        def add(hs, src):
            for h in hs:
                if h not in seen:
                    seen.add(h)
                    herbs.append((h, src))

        for s in m.get("skills", []):
            if s["branch"] == "formula":
                add(s["core"], s["id"])
                for mod in s.get("mods", []):
                    if all(t in text for t in mod.get("if", [])):
                        add(mod.get("add", []), s["id"])
            elif s["branch"] == "modification":
                add(s["add"], s["id"])
            elif s["branch"] == "free":
                mm = re.search(r"常用药：([^；。]+)", s.get("content", ""))
                if mm:
                    add([x for x in mm.group(1).split("、") if x], s["id"])
        if not herbs and m.get("cases"):
            add(m["cases"][0]["herbs"], "case")
        if not herbs:
            add(m.get("fallback", ["甘草"]), "self")
        herbs = herbs[:15]
        if m.get("style") == "direct":
            return {"herbs": [h for h, _ in herbs]}
        used = sorted({src for _, src in herbs if src not in ("case", "self")})
        return {
            "syndrome": "（模拟）",
            "principle": "（模拟）",
            "used_skills": used,
            "herbs": [{"name": h, "role": "", "source": src, "targets": []} for h, src in herbs],
        }

    @staticmethod
    def _h_reflect(m: dict) -> dict:
        gt = m.get("gt", [])
        cues = [t for t in (m.get("symptoms") or []) if t] or ([m.get("text", "")[:2]] if m.get("text") else [])
        analysis = {"syndrome": "（模拟）", "principle": "（模拟）", "base_formula": None, "modules": [{"herbs": gt[:3], "function": "（模拟）", "targets": cues[:2]}]}
        lesson = f"见{cues[0]}可考虑{gt[0]}" if cues and gt else "（模拟经验）"
        if m.get("hindsight"):
            return {"gt_analysis": analysis, "lesson": lesson}
        return {
            "gt_analysis": analysis,
            "missed": [{"herb": h, "reason": "（模拟）", "cue": cues[:2]} for h in m.get("fn", [])[:4]],
            "extra": [{"herb": h, "reason": "（模拟）"} for h in m.get("fp", [])[:4]],
            "skill_feedback": [],
            "lesson": lesson,
        }

    @staticmethod
    def _h_induce(m: dict) -> dict:
        cases = m.get("cases", [])
        terms = m.get("terms", [])
        core = m.get("core", [])
        if not cases or not terms or not core:
            return {"operations": [{"op": "NONE", "reason": "证据不足（模拟）"}]}
        if not m.get("structured", True):
            return {"operations": [{"op": "CREATE", "skill": {"branch": "task", "title": f"模拟-{core[0]}", "when": "、".join(terms[:3]), "content": f"处方思路：（模拟）；常用药：{'、'.join(core[:8])}"}}]}
        mods = [{"if": [d["term"]], "add": [d["herb"]]} for d in m.get("diff", [])[:3]]
        ops = [
            {
                "op": "CREATE",
                "skill": {
                    "branch": "formula",
                    "name": f"模拟方证-{core[0]}",
                    "trigger": {"required": terms[:1], "optional": terms[1:4], "exclude": []},
                    "pathogenesis": "（模拟）",
                    "principle": "（模拟）",
                    "core_herbs": [{"name": h, "role": "君" if i == 0 else "臣"} for i, h in enumerate(core[:8])],
                    "modifications": mods,
                    "note": "",
                },
            }
        ]
        if m.get("related") and mods:
            ops.append({"op": "PATCH", "id": m["related"][0], "patch": {"add_modifications": mods[:1]}})
        return {"operations": ops}

    @staticmethod
    def _h_split(m: dict) -> dict:
        t = m.get("sep_term")
        orig = m.get("required", [])
        new = None
        if t and len(m.get("b_core", [])) >= 2:
            new = {"branch": "formula", "name": f"模拟拆分-{t}", "trigger": {"required": [t], "optional": [], "exclude": []}, "pathogenesis": "", "principle": "", "core_herbs": [{"name": h, "role": ""} for h in m["b_core"][:8]], "modifications": [], "note": ""}
        return {"narrow": {"trigger": {"required": orig, "optional": [], "exclude": [t] if t else []}}, "new_skill": new}

    @staticmethod
    def _h_expel(m: dict) -> dict:
        ops = []
        items = m.get("items", [])
        if items and items[0].get("fn") and items[0].get("gt"):
            ops.append({"op": "ADD", "text": f"（模拟）此类病例勿漏{items[0]['fn'][0]}"})
        if m.get("n_insights", 0) > 0:
            ops.append({"op": "AGREE", "id": 1})
        return {"operations": ops}

    @staticmethod
    def _h_dc(m: dict) -> dict:
        prev = m.get("cheatsheet", "")
        items = m.get("items", [])
        line = f"（模拟）见{items[0]['text'][:6]}可用{'、'.join(items[0]['gt'][:3])}" if items else ""
        return {"cheatsheet": (prev + "\n" + line).strip()[-2000:]}
