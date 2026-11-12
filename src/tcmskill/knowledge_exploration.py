"""Working skills in ordinary multi-solution training, without maturity gates."""
from __future__ import annotations

import copy
import json

from .knowledge import digest, raw_text
from .knowledge_cultivation import draft_context, working_entries
from .knowledge_lifecycle import group_key
from .skillbook import SkillBook


def plan_exploration(exp, groups, fresh_ids):
    entries = working_entries(exp, include_pending=True)
    work_book = SkillBook()
    work_book.rules = [exp._hypothesis_rule(e) for e in entries]
    by_hid = {e["id"]: e for e in entries}
    views = set(exp.kc.get("exploration_views", ["syndrome", "integrated"]))
    plan = {"cases": [], "screened": []}
    for group in groups:
        case = exp.res.cases[group["cid"]]
        text = raw_text(case)
        before = exp.router.resolve(exp.book, case)
        fresh = case.cid in fresh_ids
        packets = (exp.retriever.packets(text, case.cid) if fresh
                   else [c["packet"] for c in group["candidates"]])
        slots = [{"packet": copy.deepcopy(p), "generate": fresh, "working_skill_id": None} for p in packets]
        positions = [i for i, p in enumerate(packets) if p["view"] in views]
        if len(positions) == len(slots):
            positions = positions[1:]  # Always retain at least one non-exploratory solution.
        applicable = []
        if positions:
            for hit in exp.router.retrieve(work_book, text):
                entry = by_hid[hit["id"]]
                proposal, route, decision = draft_context(exp, entry, case, before)
                plan["screened"].append({"hypothesis_id": entry["id"], "cid": case.cid, "applicability": decision})
                if proposal is not None:
                    applicable.append((entry, proposal, route))
                if len(applicable) == len(positions):
                    break
        if applicable:
            pool = exp.retriever.pool(text, case.cid)
            for offset, index in enumerate(positions):
                entry, proposal, route = applicable[offset % len(applicable)]
                old = packets[index]
                packet = exp.retriever.packet(pool, old["view"], old["variant"])
                slots[index].update(packet=packet, generate=True, working_skill_id=entry["id"],
                                    definition_key=entry["definition_key"], maturity=entry["status"],
                                    proposal=proposal.state_dict(), activation=route,
                                    prompt_hash=exp.policy.prompt_hash(proposal, case, {"_packet": packet, "_activation": route}))
        plan["cases"].append({"cid": case.cid, "activation": before, "slots": slots})
    return plan


def generate_exploration(exp, groups, fresh_ids, save_json):
    identity = {"protocol_id": exp.protocol_id, "step": exp.step + 1, "base": exp.book.signature(),
                "cases": [[g["cid"], digest(g["raw_text"])] for g in groups], "fresh_ids": sorted(fresh_ids),
                "library_signature": digest(exp.libraries.snapshot()), "seen_ids": list(exp.seen_ids)}
    path = exp.run_dir / "exploration_plans" / (digest(identity) + ".json")
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["identity"] != identity or digest(saved["plan"]) != saved["plan_signature"]:
            raise ValueError("Frozen exploration plan changed")
        plan = saved["plan"]
    else:
        plan = plan_exploration(exp, groups, set(fresh_ids))
        save_json(path, {"identity": identity, "plan": plan, "plan_signature": digest(plan)})
    by_id = {g["cid"]: g for g in groups}
    jobs = {}
    for item in plan["cases"]:
        group = by_id[item["cid"]]
        if item["cid"] in fresh_ids:
            group["candidates"] = [None] * len(item["slots"])
        for index, slot in enumerate(item["slots"]):
            if not slot["generate"]:
                group["candidates"][index]["candidate_provenance"] = "historical_replay"
                continue
            book = SkillBook.from_state(slot["proposal"]) if slot["working_skill_id"] else exp.book
            bucket = jobs.setdefault(book.signature(), {"book": book, "items": []})
            bucket["items"].append((item, index, slot))
    for bucket in jobs.values():
        items = bucket["items"]
        cases = [exp.res.cases[item["cid"]] for item, _, _ in items]
        packets = [slot["packet"] for _, _, slot in items]
        routes = {item["cid"]: slot.get("activation", item["activation"]) for item, _, slot in items}
        outputs = exp.policy.run(bucket["book"], cases, packets, exp.agent, "knowledge_solutions", activations=routes)
        for (item, index, slot), output in zip(items, outputs):
            candidate = {"packet": slot["packet"], "output": output, "candidate_provenance": "current_policy"}
            if slot["working_skill_id"]:
                candidate.update(working_skill_id=slot["working_skill_id"], working_definition_key=slot["definition_key"],
                                 working_maturity=slot["maturity"], exploration_prompt=slot["prompt_hash"],
                                 exploration_plan=str(path.relative_to(exp.run_dir)))
            by_id[item["cid"]]["candidates"][index] = candidate
    for group in groups:
        if group.get("provenance", {}).get("kind") == "replay" and any(c.get("working_skill_id") for c in group["candidates"]):
            group["provenance"].update(kind="mixed_replay_exploration", policy="historical baseline candidates and fresh working-skill candidates")
    for item in plan["screened"]:
        entry = exp.libraries.get(item["hypothesis_id"])
        text = raw_text(exp.res.cases[item["cid"]])
        entry.setdefault("exploration_screenings", {})[group_key(text)] = {
            "cid": item["cid"], "step": exp.step + 1, "case_signature": digest(text),
            "applicability": copy.deepcopy(item["applicability"])}
    return {"plan_path": str(path.relative_to(exp.run_dir)), "screened": len(plan["screened"]),
            "candidates": sum(bool(c.get("working_skill_id")) for g in groups for c in g["candidates"])}


def record_exploration(exp, groups):
    for group in groups:
        by_hid = {}
        for index, candidate in enumerate(group["candidates"]):
            if candidate.get("working_skill_id"):
                by_hid.setdefault(candidate["working_skill_id"], []).append(index)
        group["explorations"] = []
        for hid, indices in by_hid.items():
            entry = exp.libraries.get(hid)
            comparisons = [p for p in group["preferences"] if (p["better"] in indices) != (p["worse"] in indices)
                           and not any(group["candidates"][p[k]].get("practice_arm") for k in ("better", "worse"))]
            observation = {"cid": group["cid"], "case_signature": digest(group["raw_text"]), "step": exp.step + 1,
                           "hypothesis_id": hid, "definition_key": entry["definition_key"], "candidate_indices": indices,
                           "outcomes": [{"index": i, "valid": bool(group["candidates"][i]["output"]["ok"]),
                                         "scores": copy.deepcopy(group["candidates"][i]["assessment"]["scores"])} for i in indices],
                           "comparisons": copy.deepcopy(comparisons),
                           "provenance": "current_policy_multi_solution_contrast; not_matched_or_causal"}
            group["explorations"].append(observation)
            entry.setdefault("exploration_observations", {}).setdefault(group_key(group["raw_text"]), copy.deepcopy(observation))
            entry["updated_step"] = exp.step + 1
