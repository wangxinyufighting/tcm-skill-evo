"""Label-blind, matched training trials of immature skill versions."""
from __future__ import annotations

import copy
import json

from .knowledge import KnowledgeBase, digest, raw_text
from .knowledge_lifecycle import group_key
from .knowledge_routing import routing_text
from .skillbook import SkillBook

GUARDS = ("risk_flags", "citation_errors", "anchor_errors", "oov_count")


def working_entries(exp, include_pending=False):
    statuses = {"hypothesis", "pending"} if include_pending else {"hypothesis"}
    runnable = [e for e in exp.libraries.entries if e["status"] in {"hypothesis", "pending"} and not e.get("archived")
                  and e["normalized"] is not None and e["normalized"]["op"] in ("add", "replace")
                  and (e["definition_review"] or {}).get("ok")
                  and (e["normalized"]["op"] != "replace" or exp._target_signature(e["normalized"]) == e["target_signature"])]
    # Prefer a runnable newer version, but keep ancestors intact for comparison and fallback.
    shadowed = {p for e in runnable if not e["gate_failed"] for p in e.get("parents", [])}
    return [e for e in runnable if e["status"] in statuses and e["id"] not in shadowed]


def draft_context(exp, entry, case, before, decision=None):
    rule = exp._hypothesis_rule(entry)
    decision = decision or exp.router.judge(rule, raw_text(case), case.cid)
    if decision["decision"] != "applicable":
        return None, None, decision
    proposal = copy.deepcopy(exp.book)
    replaced = entry["normalized"].get("id") if entry["normalized"]["op"] == "replace" else None
    proposal.rules = [r for r in proposal.rules if r.id != replaced] + [rule]
    after = {"book_signature": proposal.signature(), "case_signature": digest(raw_text(case)),
             "retrieved": [r for r in before["retrieved"] if r["id"] != replaced] + [{"id": entry["id"], "score": None}],
             "decisions": [d for d in before["decisions"] if d["rule_id"] != replaced] + [decision],
             "applied_ids": [rid for rid in before["applied_ids"] if rid != replaced] + [entry["id"]]}
    return proposal, after, decision


def plan_practice(exp, groups):
    unique = {}
    for group in groups:
        unique.setdefault(group_key(group["raw_text"]), (group["cid"], group["raw_text"]))
    cases = list(unique.values())
    if not cases:
        return {"selected": [], "screened": []}
    index = KnowledgeBase.make_index(cases)
    choices = []
    for entry in working_entries(exp):
        rule = exp._hypothesis_rule(entry)
        scores = dict(index.query([routing_text(rule)], len(cases))[0])
        for cid, text in cases:
            if group_key(text) in entry.get("practice_observations", {}) or group_key(text) in entry.get("practice_screenings", {}):
                continue
            score = scores.get(cid, 0.0)
            if score > 0 or rule.details["kind"] == "principle":
                choices.append((entry.get("last_practice_step", 0), -score, entry["created_step"], entry["id"], cid))
    selected, screened = [], []
    for _, _, _, hid, cid in sorted(choices)[:int(exp.kc.get("hypothesis_trial_scan_limit", 8))]:
        if len(selected) >= int(exp.kc.get("hypothesis_trial_budget", 2)):
            break
        entry, case = exp.libraries.get(hid), exp.res.cases[cid]
        rule = exp._hypothesis_rule(entry)
        decision = exp.router.judge(rule, raw_text(case), cid)
        screened.append({"hypothesis_id": hid, "cid": cid, "decision": decision})
        if decision["decision"] != "applicable":
            continue
        packet = exp.retriever.packet(exp.retriever.pool(raw_text(case), cid), exp.kc.eval_view)
        before = exp.router.resolve(exp.book, case)
        proposal, after, _ = draft_context(exp, entry, case, before, decision)
        control_prompt = exp.policy.prompt_hash(exp.book, case, {"_packet": packet, "_activation": before})
        treatment_prompt = exp.policy.prompt_hash(proposal, case, {"_packet": packet, "_activation": after})
        if control_prompt == treatment_prompt:
            continue
        selected.append({"hypothesis_id": hid, "definition_key": entry["definition_key"], "cid": cid,
                         "case_signature": digest(raw_text(case)), "packet": packet, "proposal": proposal.state_dict(),
                         "before": before, "after": after, "control_prompt": control_prompt, "treatment_prompt": treatment_prompt})
    return {"selected": selected, "screened": screened}


def generate_practice(exp, groups, save_json):
    identity = {"protocol_id": exp.protocol_id, "step": exp.step + 1, "base": exp.book.signature(),
                "cases": [[g["cid"], digest(g["raw_text"])] for g in groups],
                "library_signature": digest(exp.libraries.snapshot()), "seen_ids": exp.seen_ids}
    path = exp.run_dir / "practice_plans" / (digest(identity) + ".json")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"] != identity or digest(saved["plan"]) != saved["plan_signature"]:
            raise ValueError("Frozen practice plan changed")
        plan = saved["plan"]
    else:
        plan = plan_practice(exp, groups)
        save_json(path, {"identity": identity, "plan": plan, "plan_signature": digest(plan)})
    by_id = {g["cid"]: g for g in groups}
    for trial in plan["selected"]:
        case, packet = exp.res.cases[trial["cid"]], trial["packet"]
        indices = []
        for arm, book, route in (("control", exp.book, trial["before"]),
                                 ("treatment", SkillBook.from_state(trial["proposal"]), trial["after"])):
            output = exp.policy.run(book, [case], [packet], exp.agent, "practice_" + arm,
                                    activations={case.cid: route})[0]
            candidates = by_id[case.cid]["candidates"]
            indices.append(len(candidates))
            candidates.append({"packet": packet, "output": output, "practice_arm": arm, "hypothesis_id": trial["hypothesis_id"]})
        record = {k: copy.deepcopy(trial[k]) for k in ("hypothesis_id", "definition_key", "cid", "case_signature", "control_prompt", "treatment_prompt")}
        record.update(control_index=indices[0], treatment_index=indices[1], step=exp.step + 1,
                      control_packet_id=packet["id"], treatment_packet_id=packet["id"], base_book_signature=exp.book.signature(),
                      plan_path=str(path.relative_to(exp.run_dir)), provenance="fresh_matched_practice; not_independent_validation")
        by_id[case.cid].setdefault("practice_trials", []).append(record)
    for item in plan["screened"]:
        entry = exp.libraries.get(item["hypothesis_id"])
        text = raw_text(exp.res.cases[item["cid"]])
        entry.setdefault("practice_screenings", {})[group_key(text)] = {
            "cid": item["cid"], "step": exp.step + 1, "case_signature": digest(text),
            "applicability": copy.deepcopy(item["decision"])}
        entry["last_practice_step"] = exp.step + 1
    return {"plan_path": str(path.relative_to(exp.run_dir)), "screened": len(plan["screened"]), "pairs": len(plan["selected"])}


def record_practice(exp, groups):
    for group in groups:
        for trial in group.get("practice_trials", []):
            before, after = [group["candidates"][trial[key]] for key in ("control_index", "treatment_index")]
            scores_before, scores_after = before["assessment"]["scores"], after["assessment"]["scores"]
            trial.update(scores_before=copy.deepcopy(scores_before), scores_after=copy.deepcopy(scores_after),
                         control_ok=bool(before["output"]["ok"]), treatment_ok=bool(after["output"]["ok"]),
                         valid_comparison=bool(before["output"]["ok"]),
                         delta=scores_after["f1"] - scores_before["f1"],
                         guard_deltas={k: scores_after[k] - scores_before[k] for k in GUARDS},
                         case_quotes=[group["raw_text"][:200]])
            entry = exp.libraries.get(trial["hypothesis_id"])
            key = group_key(group["raw_text"])
            if key in entry["practice_observations"]:
                raise ValueError("Repeated description group cannot multiply practice evidence")
            entry["practice_observations"][key] = copy.deepcopy(trial)
            entry["updated_step"] = exp.step + 1
