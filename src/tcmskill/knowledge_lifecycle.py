"""Persistent hypothesis/pending libraries and training-only contrast assessments."""
from __future__ import annotations

import copy
import json

from .decision import text_key
from .knowledge import digest, raw_text
from .knowledge_routing import field_text, routing_skill
from .llm import LLMRequestError

LIBRARY_SCHEMA = "skill-libraries-v1"
_EVIDENCE_FIELDS = {"origin_cases", "supported_cases", "support_evidence", "counterexamples", "counterexample_notes",
                    "knowledge_ids", "evidence_basis", "evidence_status", "schema", "practice_cases", "practice_summary",
                    "exploration_cases", "exploration_summary"}


def definition_key(edit, target_signature=None):
    details = edit.get("details")
    core = {k: v for k, v in details.items() if k not in _EVIDENCE_FIELDS} if isinstance(details, dict) else details
    return digest({"op": edit.get("op"), "id": edit.get("id", ""), "text": edit.get("text"),
                   "scope": edit.get("scope", []), "details": core, "target": target_signature})


def group_key(text):
    return digest(text_key(text))


def positive_observations(entry):
    return [o for o in entry["observations"].values()
            if o["applicability"]["decision"] == "applicable" and o.get("contrast", {}).get("decision") == "supports"]


def practice_summary(entry):
    records = list(entry.get("practice_observations", {}).values())
    valid = [r for r in records if r["valid_comparison"]]
    return {"attempts": len(records), "valid_groups": len(valid),
            "invalid_treatments": sum(not r["treatment_ok"] for r in records),
            "wins": sum(r["delta"] > 0 for r in valid), "losses": sum(r["delta"] < 0 for r in valid),
            "ties": sum(r["delta"] == 0 for r in valid),
            "mean_delta": sum(r["delta"] for r in valid) / len(valid) if valid else None,
            "guards_ok": all(sum(r["guard_deltas"][k] for r in valid) <= 0
                             for k in ("risk_flags", "citation_errors", "anchor_errors", "oov_count"))}


def practice_ready(entry, min_support):
    summary = practice_summary(entry)
    return (summary["valid_groups"] >= min_support and summary["wins"] > 0
            and summary["mean_delta"] >= 0 and summary["guards_ok"])


def exploration_summary(entry):
    records = list(entry.get("exploration_observations", {}).values())
    return {"groups": len(records), "valid_groups": sum(any(o["valid"] for o in r["outcomes"]) for r in records),
            "better_groups": sum(any(p["better"] in r["candidate_indices"] for p in r["comparisons"]) for r in records),
            "worse_groups": sum(any(p["worse"] in r["candidate_indices"] for p in r["comparisons"]) for r in records),
            "no_preference_groups": sum(not r["comparisons"] for r in records),
            "invalid_outputs": sum(not o["valid"] for r in records for o in r["outcomes"])}


def cultivation_ready(entry, min_support):
    if not entry.get("exploration_observations"):
        return practice_ready(entry, min_support)
    exploration = entry["exploration_observations"]
    paired = entry.get("practice_observations", {})
    groups = {k for k, r in exploration.items() if any(o["valid"] for o in r["outcomes"])}
    groups.update(k for k, r in paired.items() if r["valid_comparison"])
    practice = practice_summary(entry)
    # Contrasts only nominate a frozen version for independent testing; they do not prove a skill effect.
    return (len(groups) >= min_support and (exploration_summary(entry)["better_groups"] > 0 or practice["wins"] > 0)
            and (practice["mean_delta"] is None or practice["mean_delta"] >= 0) and practice["guards_ok"])


class SkillLibraries:
    def __init__(self, state=None):
        state = state or {"schema": LIBRARY_SCHEMA, "next_id": 1, "entries": []}
        if (not isinstance(state, dict) or state.get("schema") != LIBRARY_SCHEMA
                or type(state.get("next_id")) is not int or state["next_id"] < 1 or not isinstance(state.get("entries"), list)):
            raise ValueError("Invalid skill library schema")
        self.next_id = state["next_id"]
        self.entries = copy.deepcopy(state["entries"])
        self.events = []

    def snapshot(self):
        return {"schema": LIBRARY_SCHEMA, "next_id": self.next_id, "entries": copy.deepcopy(self.entries)}

    def get(self, hid):
        return next((e for e in self.entries if e["id"] == hid), None)

    def counts(self, book):
        return {"hypotheses": sum(e["status"] == "hypothesis" for e in self.entries),
                "pending_skills": sum(e["status"] == "pending" for e in self.entries), "formal_skills": len(book.rules),
                "hypotheses_needing_revision": sum(e["status"] == "hypothesis" and e["needs_revision"] for e in self.entries),
                "archived_hypotheses": sum(e.get("archived", False) for e in self.entries)}

    def move(self, entry, status, reason, step):
        if status not in {"hypothesis", "pending", "formal"}:
            raise ValueError("Invalid skill library status")
        if entry["status"] != status or entry["reason"] != reason:
            event = {"id": entry["id"], "step": step, "from": entry["status"], "to": status, "reason": reason}
            entry["history"].append(event)
            self.events.append(copy.deepcopy(event))
        entry.update(status=status, reason=reason, updated_step=step)

    def remember(self, edit, normalized, origin_cases, step, reason=None, target_signature=None):
        key = definition_key(normalized or edit, target_signature)
        entry = next((e for e in self.entries if e["definition_key"] == key), None)
        fresh = entry is None
        if fresh:
            entry = {"id": f"H{self.next_id}", "definition_key": key, "edit": copy.deepcopy(edit), "normalized": None,
                     "target_signature": target_signature, "status": "hypothesis", "reason": "new_hypothesis",
                     "created_step": step, "updated_step": step, "origin_cases": [], "versions": [],
                     "observations": {}, "definition_review": None, "needs_revision": False, "gate_failed": False,
                     "last_review_step": 0, "last_gate_step": 0, "gate_history": [], "gate_scan": None,
                     "practice_observations": {}, "practice_screenings": {}, "last_practice_step": 0, "parents": [],
                     "exploration_observations": {}, "exploration_screenings": {},
                     "formal_rule_id": None, "history": []}
            self.next_id += 1
            self.entries.append(entry)
        entry["origin_cases"] = sorted(set(entry["origin_cases"]) | set(origin_cases))
        entry["versions"].append({"step": step, "edit": copy.deepcopy(edit), "validation_reason": reason,
                                  "origin_cases": list(origin_cases)})
        if (normalized is not None and entry["status"] not in {"formal", "pending"} and not entry["gate_failed"]
                and not entry.get("archived") and not entry.get("practice_observations") and not entry.get("exploration_observations")):
            previous = entry["normalized"]
            value = copy.deepcopy(normalized)
            if previous is not None and value["op"] != "delete":
                value["details"]["knowledge_ids"] = list(dict.fromkeys(previous["details"]["knowledge_ids"] + value["details"]["knowledge_ids"]))
                value["details"]["evidence_basis"] = "case_contrast_and_knowledge" if value["details"]["knowledge_ids"] else "case_contrast"
            if previous != value:
                entry["normalized"] = value
                entry["definition_review"] = None
                entry["needs_revision"] = False
            entry["edit"] = copy.deepcopy(edit)
        if reason and entry["normalized"] is None:
            entry["needs_revision"] = True
            self.move(entry, "hypothesis", reason, step)
        elif fresh:
            self.move(entry, "hypothesis", "awaiting_evidence", step)
        return entry, fresh

    def retire(self, rid, step, reason, except_id=None):
        for entry in self.entries:
            if entry["status"] == "formal" and entry["formal_rule_id"] == rid and entry["id"] != except_id:
                entry["needs_revision"] = True
                entry["archived"] = True
                self.move(entry, "hypothesis", reason, step)

    def summaries(self, status, limit=15):
        entries = [e for e in self.entries if e["status"] == status and not e.get("archived")]
        entries.sort(key=lambda e: (-e["updated_step"], e["id"]))
        return [{"id": e["id"], "text": str(e["edit"].get("text", ""))[:200], "reason": e["reason"],
                 "proposal": copy.deepcopy(e["normalized"] or e["edit"]),
                 "revision_reason": e.get("revision_reason", ""),
                 "needs_revision": e["needs_revision"], "origin_cases": e["origin_cases"][-8:],
                 "verified_support_cases": [o["cid"] for o in positive_observations(e)][:8],
                 "verified_support_count": len(positive_observations(e)),
                 "practice_summary": practice_summary(e), "parents": e.get("parents", []),
                 "exploration_summary": exploration_summary(e),
                 "exploration_feedback": sorted(e.get("exploration_observations", {}).values(), key=lambda r: -r["step"])[:3],
                 "practice_feedback": sorted(e.get("practice_observations", {}).values(),
                                             key=lambda r: (r["delta"], -r["step"]))[:4],
                 "applicability_feedback": [{"cid": s["cid"], "decision": s["applicability"]["decision"],
                                              "reason": s["applicability"].get("reported_reason", s["applicability"].get("reason")),
                                              "requirements": s["applicability"].get("requirements", [])}
                                             for s in sorted({**e.get("practice_screenings", {}), **e.get("exploration_screenings", {})}.values(),
                                                             key=lambda s: -s["step"])
                                             if s["applicability"]["decision"] != "applicable"][:2],
                 "unknown_cases": len({k for k, o in {**e["observations"], **e.get("practice_screenings", {}), **e.get("exploration_screenings", {})}.items()
                                       if o["applicability"]["decision"] == "uncertain"}),
                 "opposing_cases": sum(o.get("contrast", {}).get("decision") == "opposes" for o in e["observations"].values()),
                 "definition_review": e["definition_review"] if e["needs_revision"] else None}
                for e in entries[:limit]]

    def validate(self, res, seen_ids, book, gate_pool, min_support=1, require_formal_registry=False, cultivate=False):
        ids, keys, formal_ids = set(), set(), set()
        seen, gate_pool = set(seen_ids), set(gate_pool)
        for entry in self.entries:
            hid = entry["id"]
            if (not isinstance(hid, str) or not hid.startswith("H") or not hid[1:].isdigit()
                    or not 0 < int(hid[1:]) < self.next_id
                    or hid in ids or entry["definition_key"] in keys or entry["status"] not in {"hypothesis", "pending", "formal"}
                    or not set(entry["origin_cases"]) <= seen):
                raise ValueError("Invalid hypothesis library identity or origin cases")
            if entry["definition_key"] != definition_key(entry["normalized"] or entry["edit"], entry["target_signature"]):
                raise ValueError("Skill library definition changed without a revision")
            ids.add(entry["id"])
            keys.add(entry["definition_key"])
            if entry["status"] == "formal" and book.get(entry["formal_rule_id"]) is None:
                raise ValueError("Formal library reference is not in the active skill book")
            if entry["status"] == "formal":
                rid = entry["formal_rule_id"]
                if rid in formal_ids or not entry["gate_history"] or entry["gate_history"][-1]["reason"] != "accepted":
                    raise ValueError("Formal skill must have its own accepted gate history")
                formal_ids.add(rid)
                rule = book.get(rid)
                actual = {**entry["normalized"], "text": rule.text, "scope": rule.scope, "details": rule.details}
                if definition_key(actual, entry["target_signature"]) != entry["definition_key"]:
                    raise ValueError("Formal skill differs from its verified library definition")
            if entry["status"] in {"pending", "formal"}:
                if entry["normalized"] is None or entry["needs_revision"] or entry["gate_failed"] or entry.get("archived"):
                    raise ValueError("Invalid pending or formal skill state")
                if entry["normalized"]["op"] != "delete" and (
                    not (entry["definition_review"] or {}).get("ok") or
                    (not cultivation_ready(entry, min_support) if cultivate else len(positive_observations(entry)) < min_support)
                ):
                    raise ValueError("Pending and formal skills require verified independent support")
            for key, trial in entry.get("practice_observations", {}).items():
                cid = trial["cid"]
                if (cid not in seen or res.cases[cid].split != "train" or key != group_key(raw_text(res.cases[cid]))
                        or trial["case_signature"] != digest(raw_text(res.cases[cid]))
                        or trial["definition_key"] != entry["definition_key"]
                        or trial["control_packet_id"] != trial["treatment_packet_id"]
                        or trial["control_prompt"] == trial["treatment_prompt"]
                        or trial["delta"] != trial["scores_after"]["f1"] - trial["scores_before"]["f1"]):
                    raise ValueError("Invalid matched practice evidence or uncommitted case")
                if trial["valid_comparison"] != trial["control_ok"]:
                    raise ValueError("Invalid practice comparison validity")
                if any(trial["guard_deltas"][k] != trial["scores_after"][k] - trial["scores_before"][k]
                       for k in ("risk_flags", "citation_errors", "anchor_errors", "oov_count")):
                    raise ValueError("Invalid practice guard differences")
            for key, observation in entry.get("exploration_observations", {}).items():
                cid, indices, outcomes = observation["cid"], observation["candidate_indices"], observation["outcomes"]
                if (cid not in seen or res.cases[cid].split != "train" or key != group_key(raw_text(res.cases[cid]))
                        or observation["case_signature"] != digest(raw_text(res.cases[cid]))
                        or observation["definition_key"] != entry["definition_key"] or observation["hypothesis_id"] != hid
                        or not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices)
                        or [o["index"] for o in outcomes] != indices or any(type(o["valid"]) is not bool for o in outcomes)
                        or any((p["better"] in indices) == (p["worse"] in indices) for p in observation["comparisons"])):
                    raise ValueError("Invalid working-skill exploration evidence")
            for key, screen in [*entry.get("practice_screenings", {}).items(), *entry.get("exploration_screenings", {}).items()]:
                cid = screen["cid"]
                if (cid not in seen or res.cases[cid].split != "train" or key != group_key(raw_text(res.cases[cid]))
                        or screen["case_signature"] != digest(raw_text(res.cases[cid]))
                        or screen["applicability"]["decision"] not in {"applicable", "uncertain", "not_applicable"}):
                    raise ValueError("Invalid practice applicability record or uncommitted case")
            for key, observation in entry["observations"].items():
                cid = observation["cid"]
                if (cid not in seen or res.cases[cid].split != "train" or key != group_key(raw_text(res.cases[cid]))
                        or observation["case_signature"] != digest(raw_text(res.cases[cid]))
                        or observation["applicability"]["decision"] not in {"applicable", "uncertain", "not_applicable"}
                        or observation.get("contrast", {}).get("decision") not in {None, "supports", "opposes", "unclear"}):
                    raise ValueError("Skill evidence must come from committed practice description groups")
                if observation.get("contrast", {}).get("decision") == "supports" and observation["applicability"]["decision"] != "applicable":
                    raise ValueError("Unknown applicability cannot count as skill support")
                contrast = observation.get("contrast") or {}
                if contrast.get("decision") in {"supports", "opposes"}:
                    pairs, quotes = contrast.get("pairs", []), contrast.get("case_quotes", [])
                    allowed = {(p["better"], p["worse"]) for p in observation["preferences"]}
                    if (not pairs or not quotes or contrast.get("validation_errors")
                            or any(not isinstance(q, str) or not q.strip() or q not in raw_text(res.cases[cid]) for q in quotes)
                            or any((p["better"], p["worse"]) not in allowed for p in pairs)):
                        raise ValueError("Skill support requires valid quotes and observed preference pairs")
            scan = entry.get("gate_scan") or {}
            if not set(scan.get("examined_ids", []) + scan.get("affected_ids", [])) <= gate_pool:
                raise ValueError("Invalid pending skill gate scan")
        if require_formal_registry and formal_ids != {r.id for r in book.rules}:
            raise ValueError("Formal skill book must match the verified library registry")
        for entry in self.entries:
            if any(parent not in ids or int(parent[1:]) >= int(entry["id"][1:]) for parent in entry.get("parents", [])):
                raise ValueError("Skill lineage must refer to older library versions")


CONTRAST_SYSTEM = """你审核一条候选技能是否得到当前训练病例的组内处方对比支持，不判断新病例适用性，不改写技能。
病例及其候选、生成时理由、知识包、多维分数均来自已揭晓的练习数据。只能使用输入的 preferences 优劣对。
supports：高分与低分候选之间的具体策略差异支持技能；opposes：差异方向反而与技能相反；unclear：无可归因差异或证据不足。
病例满足适用条件、某候选 F1 高、或引用了相同术语，都不足以自动判 supports；需要说明技能与实际候选差异的联系。
这只是观察性对比，不证明因果或临床真理。不要把未提到的表现补造为患者事实，不把知识缺失判为用药错误。
监督目标就是 reference_f1 与知识约束，不得仅因不是临床/因果证明就降为 unclear，也不要求候选显式执行过后来才总结的完整技能步骤。
一个病例可以只支持局部决策差异；指出 supported_parts 与 unverified_parts，不要求它证明所有可选分支。存在具体同向差异可判 supports，但不能把未验证部分写成已验证。
只输出 JSON：{"decision":"supports|opposes|unclear","reason":"依据及局限","case_quotes":["逐字病例引用"],
"pairs":[{"better":0,"worse":1,"reason":"具体差异"}],
"supported_parts":[{"field":"decision_steps[0]","quote":"技能逐字片段","reason":"局部支持"}],
"unverified_parts":[{"field":"branches[0].actions[0]","quote":"技能逐字片段","reason":"尚无对照"}]}。
supports/opposes 必须有至少一个合法偏好对和逐字病例引用；无法判断时返回 unclear，不凑支持。"""


class ContrastSupportJudge:
    def __init__(self, cfg, llm):
        self.cfg, self.llm, self.cache = cfg, llm, {}

    def assess(self, rule, group):
        payload = {"raw_text": group["raw_text"], "skill": routing_skill(rule), "preferences": group["preferences"],
                   "candidates": [{"index": i, "herbs": c["output"]["herbs"], "rationale": c["output"].get("rationale", {}),
                                   "scores": c["assessment"]["scores"], "evidence": c["packet"]["evidence"]}
                                  for i, c in enumerate(group["candidates"])]}
        key = digest(payload)
        if key in self.cache:
            return copy.deepcopy(self.cache[key])
        result = self.llm.chat_json([{"role": "system", "content": CONTRAST_SYSTEM},
                                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                                   "knowledge_skill_support", meta={"cid": group["cid"], **payload},
                                   max_tokens=int(self.cfg.applicability_max_tokens) or None)
        if result.request_failed:
            raise LLMRequestError(f"Skill contrast support assessment failed: {result.error}")
        data = result.data if result.ok and isinstance(result.data, dict) else {}
        decision, pairs, quotes = data.get("decision"), data.get("pairs"), data.get("case_quotes")
        allowed = {(p["better"], p["worse"]) for p in group["preferences"]}
        valid = (isinstance(decision, str) and decision in {"supports", "opposes", "unclear"}
                 and isinstance(data.get("reason"), str) and bool(data["reason"].strip())
                 and isinstance(quotes, list) and all(isinstance(q, str) and q.strip() and q in group["raw_text"] for q in quotes)
                 and isinstance(pairs, list) and all(isinstance(p, dict) and type(p.get("better")) is int and type(p.get("worse")) is int
                     and 0 <= p["better"] < len(group["candidates"]) and 0 <= p["worse"] < len(group["candidates"])
                     and (p["better"], p["worse"]) in allowed and isinstance(p.get("reason"), str) and bool(p["reason"].strip()) for p in pairs)
                 and (decision == "unclear" or (bool(pairs) and bool(quotes))))
        parts = {name: data.get(name, []) for name in ("supported_parts", "unverified_parts")}
        for values in parts.values():
            valid = valid and isinstance(values, list) and all(
                isinstance(v, dict) and isinstance(v.get("quote"), str) and bool(v["quote"].strip())
                and isinstance(v.get("reason"), str) and bool(v["reason"].strip())
                and field_text(payload["skill"], v.get("field")) is not None
                and v["quote"] in field_text(payload["skill"], v["field"]) for v in values)
        record = {"decision": decision if valid else "unclear", "reason": data.get("reason", result.error or "Invalid support response"),
                  **parts,
                  "case_quotes": quotes if isinstance(quotes, list) else [], "pairs": pairs if isinstance(pairs, list) else [],
                  "validation_errors": [] if valid else ["invalid_contrast_evidence"], "request_key": result.request_key,
                  "assessment_signature": key, "limitations": "LLM alignment of observed contrasts, not a causal or clinical validation"}
        self.cache[key] = record
        return copy.deepcopy(record)
