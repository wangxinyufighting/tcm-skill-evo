"""Deterministic mock LLM for smoke tests and cost estimation (`--llm mock`).

It never sees the prompt semantics; callers pass structured `meta`, and the mock produces
plausible outputs from it (e.g. it prescribes the core herbs of retrieved skills). Token usage
is estimated from the *real* prompt text, so `--llm mock` gives a realistic cost estimate.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading

from .llm import BaseLLM, LLMRequestError, LLMResult
from .utils import estimate_tokens

TYPICAL_OUT = {"prescribe": 380, "reflect": 480, "induce": 950, "split": 700, "expel": 500, "dc": 1500, "annotate": 220, "skill_rx": 160, "skill_opt": 900,
               "knowledge_rx": 800, "knowledge_opt": 1200, "knowledge_applicability": 500, "knowledge_skill_review": 400,
               "knowledge_skill_support": 500}


class MockLLM(BaseLLM):
    name = "mock"

    def __init__(self, cfg):
        super().__init__(cfg)
        self._seen = set()
        self._lock = threading.Lock()
        # Simulated transport failures (tests only): a deterministic fraction of distinct requests fails -
        # once (transient, the default) or every time (llm.mock_fail_persistent=true).
        self._fail_rate = float(cfg.llm.get("mock_fail_rate", 0.0) or 0.0)
        self._fail_persistent = bool(cfg.llm.get("mock_fail_persistent", False))
        self._failed: set[str] = set()
        # tests: optimizer answers come back truncated when the output budget is below this
        self._opt_truncate_below = int(cfg.llm.get("mock_opt_truncate_below", 0) or 0)

    def _simulated_failure(self, messages) -> bool:
        if self._fail_rate <= 0:
            return False
        h = hashlib.md5(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        if int(h[:8], 16) / 0xFFFFFFFF >= self._fail_rate:
            return False
        with self._lock:
            if h in self._failed and not self._fail_persistent:
                return False
            self._failed.add(h)
        return True

    def chat_json(self, messages, task, meta=None, max_tokens=None) -> LLMResult:
        meta = meta or {}
        if self._simulated_failure(messages):
            res = LLMResult(ok=False, model="mock", error="ConnectionError: simulated failure", request_failed=True,
                            error_chain=[{"type": "ConnectionError", "message": "simulated failure"}], max_tokens=self.output_budget(max_tokens))
            self.usage.add(task, {}, cached=False, error=True, model=self.model)
            self._log_call(task, res, meta)
            if self._require_responses:
                raise LLMRequestError(f"{task}: simulated failure")
            return res
        if task == "skill_opt" and self.output_budget(max_tokens) < self._opt_truncate_below:
            res = LLMResult(ok=False, model="mock", error="truncated_output", truncated=True, finish_reason="length",
                            max_tokens=self.output_budget(max_tokens))
            self.usage.add(task, {}, cached=False, error=True, model=self.model)
            self._log_call(task, res, meta)
            return res
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
    def _h_knowledge_skill_support(m: dict) -> dict:
        if not m["preferences"]:
            return {"decision": "unclear", "reason": "（模拟）没有组内偏好", "case_quotes": [], "pairs": []}
        pair = m["preferences"][0]
        return {"decision": "supports", "reason": "（模拟）只测试经验积累协议，不证明技能有效",
                "supported_parts": [{"field": "text", "quote": m["skill"]["text"], "reason": "（模拟）局部观察"}],
                "unverified_parts": [],
                "case_quotes": [m["raw_text"][:20]], "pairs": [{"better": pair["better"], "worse": pair["worse"],
                                                               "reason": "（模拟）组内候选差异"}]}

    @staticmethod
    def _h_knowledge_skill_review(m: dict) -> dict:
        return {"evidence_type": "procedural" if m["skill"]["kind"] == "principle" else "medical",
                "reason": "（模拟）仅测试定义审核协议", "issues": [],
                "branch_roles": [{"index": i, "role": b.get("kind", "patient"), "reason": "（模拟）分支类型"}
                                 for i, b in enumerate(m["skill"]["branches"])],
                "requirements": [{"index": i, "role": "entry_evidence", "reason": "（模拟）必要事实"}
                                 for i, _ in enumerate(m["requirements"])]}

    @staticmethod
    def _h_knowledge_applicability(m: dict) -> dict:
        # Deliberately mechanical: this tests the protocol, not semantic or clinical accuracy.
        return {"decision": "applicable", "reason": "（模拟）仅测试适用性协议，不代表语义判断",
                "requirements": [{"index": i, "status": "supported", "quotes": [m["raw_text"][:20]],
                                  "reason": "（模拟）原文锚点"} for i, _ in enumerate(m["requirements"])],
                "exclusions": [{"index": i, "status": "unknown", "quotes": [],
                                "reason": "（模拟）未确定"} for i, _ in enumerate(m["exclusions"])],
                "branches": [{"index": i, "status": "unknown", "quotes": [], "reason": "（模拟）未确定"}
                             for i, _ in enumerate(m.get("branches", []))]}

    @staticmethod
    def _h_knowledge_rx(m: dict) -> dict:
        evidence = m["packet"]["evidence"]
        herbs, references = [], {}
        for item in evidence:
            for herb in item["herbs"]:
                if herb not in herbs:
                    herbs.append(herb)
                    references[herb] = item["id"]
            if len(herbs) >= 6:
                break
        herbs = (herbs or m["fallback"])[:6]
        quote = m["text"][:20]
        return {"syndrome_hypothesis": "（模拟，不代表医学判断）", "principle": "（模拟）按原文核对知识",
                "formula_reason": "（模拟）根据当前知识包列出候选，不读取原方", "uncertainty": "模拟输出仅用于流程测试",
                "case_evidence": [quote], "used_rules": m["rule_ids"],
                "herbs": [{"name": h, "role": "模拟", "reason": "（模拟）参考所列证据，非临床结论",
                           "case_quotes": [quote], "evidence_ids": [references[h]] if h in references else []} for h in herbs]}

    @staticmethod
    def _h_knowledge_opt(m: dict) -> dict:
        text = "对照完整主治原文核查选药依据，保留未知项，不把证候资料中的表现补入病例。"
        if text in m.get("rule_texts", []) or not m.get("evidence_ids"):
            return {"analysis": "（模拟）没有新编辑", "edits": []}
        result = {"analysis": "（模拟）比较知识来源；不代表实际学到有效技能", "edits": [
            {"op": "add", "text": text, "scope": [], "details": {
                "conditions": [], "exclude": [], "pathogenesis_hypothesis": "（模拟）未确定",
                "principle": "（模拟）核查原文", "actions": ["逐项核对病例原文及知识出处"],
                "knowledge_ids": m["evidence_ids"][:2], "supported_cases": [c["cid"] for c in m["cases"]],
                "counterexamples": [], "uncertainty": "仅为模拟，等待行为验证"}}]}
        if m.get("skill_mode") == "semantic":
            details = result["edits"][0]["details"]
            details["required_conditions"] = details.pop("conditions")
            details["decision_steps"] = details.pop("actions")
            details.update(branches=[],
                kind="principle", applicability="根据完整原文核对选药依据的病例", retrieval_terms=["原文", "依据"],
                support_evidence=[{"cid": c["cid"], "case_quotes": [c["raw_text"][:20]],
                                   "reason": "（模拟）用于验证跨病例支持的结构"} for c in m["cases"]])
        return result

    @staticmethod
    def _h_prescribe(m: dict) -> dict:
        if m.get("style") == "select_edit":
            return MockLLM._select_edit(m)
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
    def _select_edit(m: dict) -> dict:
        """Base = most similar candidate; drop its last herb if it is long; add one herb from a formula skill."""
        cases = m.get("cases", [])
        if not cases:
            herbs = list(m.get("fallback", ["甘草"]))
            return {"syndrome": "（模拟）", "principle": "（模拟）", "base": "none", "fit": "", "remove": [], "add": [], "herbs": herbs}
        base = list(cases[0]["herbs"])
        remove = base[-1:] if len(base) > 6 else []
        add = []
        for s in m.get("skills", []):
            for h in s.get("core", []) + s.get("add", []):
                if h not in base:
                    add = [h]
                    break
            if add:
                break
        herbs = [h for h in base if h not in remove] + add
        return {"syndrome": "（模拟）", "principle": "（模拟）", "base": "C1", "fit": "（模拟）", "remove": remove,
                "add": [{"name": h, "cue": "（模拟）"} for h in add], "herbs": herbs}

    # (op, text, scope, why). The mock agent is helped by the copy and trim rules, unaffected by the others.
    MOCK_SKILL_EDITS = [
        ("add", "多写几味药会更保险，尽量把候选医案的药都开上", [], "#1 #2 #3"),                       # useless, proposed together with the copy rule
        ("add", "若 C1 与本例主治几乎相同（sim≥0.95），直接照抄 C1 的处方，不要加减", [], "#2 #4 #5"),
        ("add", "某一病例见到的特殊用药：酌加黄连", [], "#1"),                                         # weak support: dropped without a gate
        ("add", "处方一般不超过 8 味：只保留多数相似医案共有的药", [], "#3 #6 #7"),
        ("add", "产后病证在相似医案的基础上酌加当归、川芎以养血活血", ["产后"], "#2 #8 #9"),
        ("add", "相似度一般时改用多案投票的结果", [], "支持 #1 #2 #3；反例 #4 #5"),        # too many counter-examples: dropped
    ]

    @staticmethod
    def _h_skill_rx(m: dict) -> dict:
        """Over-prescribes (union of C1 and C2) unless its skills tell it to copy a near-identical C1 or to trim."""
        rules, ids, cases = m.get("rules", []), m.get("rule_ids", []), m.get("cases", [])
        copy_ids = [i for i, r in zip(ids, rules) if "照抄" in r]
        trim_ids = [i for i, r in zip(ids, rules) if "不超过" in r]
        used = []
        if copy_ids and cases and cases[0].get("sim", 0) >= 0.95:
            herbs, used = list(cases[0]["herbs"]), used + copy_ids
        else:
            herbs = []
            for c in cases[:2]:
                herbs += [h for h in c["herbs"] if h not in herbs]
            herbs = herbs or list(m.get("fallback", ["甘草"]))
        if trim_ids and len(herbs) > 8:
            herbs, used = herbs[:8], used + trim_ids
        return {"analysis": "（模拟）", "used_rules": used, "herbs": herbs}

    @staticmethod
    def _h_skill_opt(m: dict) -> dict:
        have = set(m.get("rule_texts", [])) | set(m.get("rejected_texts", []))
        edits = [{"op": op, "text": text, "scope": scope, "why": why} for op, text, scope, why in MockLLM.MOCK_SKILL_EDITS if text not in have]
        return {"analysis": "（模拟）" if edits else "（模拟）没有新的编辑", "edits": edits[: int(m.get("k_edit", 1))]}

    @staticmethod
    def _h_annotate(m: dict) -> dict:
        core = m.get("core", [])
        return {"name": f"模拟方证-{core[0] if core else ''}", "pathogenesis": "（模拟病机）", "principle": "（模拟治法）",
                "roles": {h: ("君" if i == 0 else "臣") for i, h in enumerate(core[:6])}, "note": "（模拟）"}

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
