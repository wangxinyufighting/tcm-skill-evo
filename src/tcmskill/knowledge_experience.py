"""Committed contrast episodes, cross-batch retrieval, and provenance-preserving replay."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .decision import text_key
from .knowledge import KnowledgeBase, digest, raw_text


def episode_text(group):
    parts = [group["raw_text"]]
    for candidate in group["candidates"]:
        rationale = candidate["output"].get("rationale", {})
        parts.extend(str(rationale.get(k, "")) for k in ("syndrome_hypothesis", "principle", "formula_reason"))
    # This bound applies only to the search index, never to the narrative sent to a model.
    return "\n".join(parts)[:16000]


class ExperiencePool:
    def __init__(self, capacity, groups=()):
        self.capacity = int(capacity)
        self.groups = {}
        self.add(groups)

    def add(self, groups):
        keys = {text_key(g["raw_text"]) for g in self.groups.values()}
        for group in groups:
            key = text_key(group["raw_text"])
            if group["cid"] not in self.groups and key not in keys:
                self.groups[group["cid"]] = copy.deepcopy(group)
                keys.add(key)
        while len(self.groups) > self.capacity:
            del self.groups[next(iter(self.groups))]

    def snapshot(self):
        return copy.deepcopy(list(self.groups.values()))

    def select(self, current, count):
        combined = dict(self.groups)
        combined.update({g["cid"]: g for g in current})
        unique = {}
        for group in combined.values():
            if group.get("preferences") or group.get("practice_trials") or group.get("explorations"):
                unique.setdefault(text_key(group["raw_text"]), group)
        candidates = list(unique.values())
        if not candidates:
            return []
        current_keys = {text_key(g["raw_text"]) for g in current}
        anchors = sorted((g for g in candidates if text_key(g["raw_text"]) in current_keys),
                         key=lambda g: (not bool(g.get("explorations") or g.get("practice_trials")), -g["f1_gap"], g["cid"]))[:max(1, count // 2)]
        if not anchors:
            anchors = sorted(candidates, key=lambda g: (-g["f1_gap"], g["cid"]))[:1]
        index = KnowledgeBase.make_index([(g["cid"], episode_text(g)) for g in candidates])
        scores = {}
        for neighbors in index.query([episode_text(g) for g in anchors], len(candidates)):
            for cid, score in neighbors:
                scores[cid] = max(scores.get(cid, 0.0), score)
        ordered = anchors + sorted(candidates, key=lambda g: (-scores.get(g["cid"], 0.0), -g["f1_gap"], g["cid"]))
        selected, seen = [], set()
        for group in ordered:
            key = text_key(group["raw_text"])
            if key in seen:
                continue
            seen.add(key)
            item = copy.deepcopy(group)
            item["experience_role"] = "current" if key in current_keys else "history"
            item["experience_retrieval_score"] = scores.get(group["cid"], 0.0)
            selected.append(item)
            if len(selected) == count:
                break
        return selected


def read_replay_sources(paths, res, kb_version, dataset_sig):
    """Import only committed practice outputs; never import gates, edits, or skills."""
    archive = {"schema": "knowledge-replay-v1", "sources": [], "groups": [], "exposed_ids": [], "duplicate_episodes": 0}
    by_id, exposed, source_paths = {}, set(), set()
    for name in paths:
        path = Path(name).resolve()
        if str(path) in source_paths:
            continue
        source_paths.add(str(path))
        protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
        state = json.loads((path / "state.json").read_text(encoding="utf-8"))
        if (protocol.get("schema") not in {f"knowledge-evolution-v{i}" for i in range(1, 8)}
                or digest(protocol) != state.get("protocol_id") or protocol.get("input_mode") != "raw_description"
                or protocol.get("kb_version") != kb_version or protocol.get("dataset_signature") != dataset_sig):
            raise ValueError(f"Replay source has incompatible data, knowledge or protocol: {path}")
        seen = state["seen_ids"]
        if seen != protocol["partition"]["practice"][:state["cursor"]]:
            raise ValueError("Replay source has an invalid committed practice prefix")
        used = state.get("used_gate_ids", protocol["partition"]["gate_pool"][:state["gate_cursor"]])
        if (len(set(used)) != len(used) or len(used) != state["gate_cursor"]
                or not set(used) <= set(protocol["partition"]["gate_pool"])):
            raise ValueError("Replay source has an invalid gate ledger")
        exposed.update(seen)
        exposed.update(used)
        exposed.update(protocol["partition"].get("replay_exposed_ids", []))
        step_hashes, committed = [], []
        for number in range(1, state["step"] + 1):
            row = json.loads((path / "steps" / f"step_{number:05d}.json").read_text(encoding="utf-8"))
            if row["step"] != number or row["seen_before"] != committed:
                raise ValueError("Replay source has inconsistent step ordering")
            step_hashes.append(digest(row))
            for group in row["groups"]:
                cid = group["cid"]
                case = res.cases.get(cid)
                if (cid not in seen or case is None or case.split != "train"
                        or group["raw_text"] != raw_text(case) or group["gold"] != list(case.herbs)
                        or not group["candidates"]):
                    raise ValueError(f"Invalid replay episode: {cid}")
                if any(c["packet"].get("kb_version") != kb_version for c in group["candidates"]):
                    raise ValueError("Replay candidate knowledge version mismatch")
                if not replay_compatible(group, row["seen_before"], res):
                    raise ValueError("Replay source contains unrevealed case evidence")
                committed.append(cid)
                origin = {"run_dir": str(path), "protocol_id": state["protocol_id"], "step": number,
                          "model": protocol["config"]["llm"]["model"], "step_signature": step_hashes[-1]}
                if cid in by_id:
                    archive["duplicate_episodes"] += 1
                    by_id[cid]["provenance"]["sources"].append(origin)
                else:
                    item = copy.deepcopy(group)
                    item["candidates"] = [c for c in item["candidates"] if not c.get("practice_arm")]
                    item.pop("practice_trials", None)
                    item.pop("explorations", None)
                    for candidate in item["candidates"]:
                        hid = candidate.pop("working_skill_id", None)
                        if hid:
                            candidate["historical_working_skill"] = {"id": hid, "protocol_id": state["protocol_id"]}
                        for field in ("working_definition_key", "working_maturity", "exploration_prompt", "exploration_plan"):
                            candidate.pop(field, None)
                        candidate["candidate_provenance"] = "historical_replay"
                    if not item["candidates"]:
                        raise ValueError("Replay source contains no ordinary practice candidates")
                    item["provenance"] = {"kind": "replay", "policy": "historical; not a current-policy evaluation", "sources": [origin]}
                    by_id[cid] = item
        if committed != seen:
            raise ValueError("Replay source steps do not reconstruct the committed memory")
        archive["sources"].append({"run_dir": str(path), "protocol_id": state["protocol_id"],
                                   "state_signature": digest(state), "steps": step_hashes})
    archive["groups"] = list(by_id.values())
    archive["exposed_ids"] = sorted(exposed)
    return archive


def replay_compatible(group, seen_ids, res):
    """Past retrieved case labels must already be revealed in the new stream."""
    seen = set(seen_ids)
    for candidate in group["candidates"]:
        for hit in candidate["packet"]["evidence"]:
            if hit["kind"] == "case":
                cid = hit.get("cid")
                if cid not in seen or text_key(raw_text(res.cases[cid])) == text_key(group["raw_text"]):
                    return False
    return True
