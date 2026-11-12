"""Auditable knowledge-view evolution with revealed-only memory and fresh gates."""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import to_plain
from .decision import text_key
from .knowledge import KnowledgeBase, KnowledgeRetriever, VIEWS, digest, raw_text
from .knowledge_cultivation import generate_practice, record_practice
from .knowledge_exploration import generate_exploration, record_exploration
from .knowledge_experience import ExperiencePool, read_replay_sources, replay_compatible
from .knowledge_lifecycle import ContrastSupportJudge, SkillLibraries, cultivation_ready, exploration_summary, group_key, positive_observations, practice_summary
from .knowledge_policy import KnowledgeJudge, KnowledgePolicy, LITERAL_OPT_SYSTEM, OPT_SYSTEM, preference_pairs, strings
from .knowledge_routing import SkillRouter, TYPED_SKILL_SCHEMA, routing_skill, routing_text
from .llm import LLMRequestError
from .skillbook import Rule, SkillBook, paired_stats
from .utils import stable_shuffle


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def dataset_signature(res):
    return digest([(cid, c.split, raw_text(c), c.herbs) for cid, c in sorted(res.cases.items())])


def validate_checkpoint(state, partition, require_used_gate_ids=False):
    cursor, gate_cursor = state["cursor"], state["gate_cursor"]
    if not isinstance(cursor, int) or not 0 <= cursor <= len(partition["practice"]):
        raise ValueError("Invalid checkpoint practice cursor")
    if state["seen_ids"] != partition["practice"][:cursor]:
        raise ValueError("Checkpoint memory is not the committed revealed practice prefix")
    if not isinstance(gate_cursor, int) or not 0 <= gate_cursor <= len(partition["gate_pool"]):
        raise ValueError("Invalid checkpoint gate cursor")
    used = state.get("used_gate_ids")
    if require_used_gate_ids and used is None:
        raise ValueError("Checkpoint requires explicit used gate groups")
    if used is not None:
        if (not isinstance(used, list) or any(not isinstance(cid, str) for cid in used)
                or len(used) != gate_cursor or len(set(used)) != len(used)
                or not set(used) <= set(partition["gate_pool"])):
            raise ValueError("Invalid checkpoint used gate groups")


def grouped_partition(res, n_practice, seed):
    held_out_keys = {text_key(raw_text(c)) for c in res.cases.values() if c.split != "train"}
    groups = {}
    excluded = []
    for cid in res.select("train"):
        case = res.cases[cid]
        key = text_key(raw_text(case))
        if key in held_out_keys:
            excluded.append(cid)
        else:
            groups.setdefault(key, []).append(cid)
    order = stable_shuffle(sorted(groups), seed)
    practice, gate = [], []
    for key in order:
        if len(practice) < n_practice:
            practice.extend(groups[key])
        else:
            # One representative per group avoids counting repeated descriptions as independent evidence.
            gate.append(groups[key][0])
    if len(practice) < n_practice:
        raise ValueError("Not enough evaluation-group-disjoint training cases for knowledge.n_practice")
    return {"practice": practice, "gate_pool": gate, "excluded_eval_overlap": excluded,
            "grouping": "punctuation-normalized desc_raw; gate representatives are unique description groups"}


def load_kb(res, cfg):
    return KnowledgeBase.load(cfg.paths.ptm_dir, res.lexicon, os.path.join(cfg.paths.kb_dir, "herb_aliases.json"),
                              os.path.join(cfg.paths.kb_dir, "incompatibility.json"))


def validate_config(cfg):
    kc = cfg.knowledge
    if not kc.views or len(set(kc.views)) != len(kc.views) or any(v not in VIEWS for v in kc.views):
        raise ValueError("knowledge.views must contain distinct supported views")
    for name in ("variants_per_view", "variant_band", "relation_k", "text_k", "case_k", "expand_herbs",
                 "max_items", "max_chars", "excerpt_chars", "per_herb_cap", "n_practice", "batch", "gate_size",
                 "min_support", "k_edit", "n_reflect", "optimizer_max_chars"):
        if int(kc[name]) <= 0:
            raise ValueError(f"knowledge.{name} must be positive")
    if int(kc.confirm_size) < 0 or kc.eval_view not in VIEWS:
        raise ValueError("Invalid confirmation size or eval view")
    if int(kc.gate_size) < 2 or (0 < int(kc.confirm_size) < 2):
        raise ValueError("Gate and enabled confirmation sets need at least two independent groups")
    if float(kc.min_gap) <= 0 or float(kc.min_delta) < 0 or float(kc.t_min) < 0:
        raise ValueError("Invalid score/gate thresholds")
    if kc.get("skill_mode", "literal") not in {"literal", "semantic"}:
        raise ValueError("knowledge.skill_mode must be literal or semantic")
    if kc.get("learning_mode", "cultivate") not in {"cultivate", "observe"}:
        raise ValueError("knowledge.learning_mode must be cultivate or observe")
    if int(kc.get("experience_capacity", 256)) < int(kc.min_support):
        raise ValueError("Experience capacity must cover the minimum independent support")
    if kc.get("skill_mode") == "semantic":
        if (int(kc.skill_retrieval_k) < 1 or not 0 <= float(kc.skill_retrieval_min_score) <= 1
                or int(kc.applicability_max_tokens) < 0
                or int(kc.gate_candidate_limit) < int(kc.gate_size) + int(kc.confirm_size)):
            raise ValueError("Invalid semantic routing or gate candidate budget")
        if int(kc.get("hypothesis_review_budget", 12)) < 1 or int(kc.get("pending_gate_budget", 1)) < 0:
            raise ValueError("Hypothesis review budget must be positive and pending gate budget nonnegative")
        if (kc.get("learning_mode", "cultivate") == "cultivate" and (int(kc.get("hypothesis_trial_budget", 0)) < 0
                or int(kc.get("hypothesis_trial_scan_limit", 8)) < int(kc.get("hypothesis_trial_budget", 0)))):
            raise ValueError("Invalid matched practice budget or screening limit")
        exploration_views = kc.get("exploration_views", ["syndrome", "integrated"])
        if (not isinstance(exploration_views, list) or any(v not in VIEWS for v in exploration_views)
                or len(set(exploration_views)) != len(exploration_views)):
            raise ValueError("Exploration views must be a list of distinct supported views")
    if not 0 < float(kc.get("knowledge_min_gap", 0.05)) <= 1:
        raise ValueError("knowledge.knowledge_min_gap must be in (0, 1]")
    if not isinstance(kc.get("replay_runs", []), list):
        raise ValueError("knowledge.replay_runs must be a list of run directories")
    if kc.get("replay_only") and not kc.get("replay_runs"):
        raise ValueError("replay_only requires at least one replay run")


def step_diagnostics(row):
    groups, trials = row["groups"], row["trials"]
    candidates = [c for g in groups for c in g["candidates"]]
    optimizer = row["optimizer"]
    data = optimizer.get("data")
    edits = data.get("edits", []) if isinstance(data, dict) else []
    selected = set(optimizer.get("selected_cases", []))
    gates = [t["gate"] for t in trials if t.get("gate") is not None]
    return {"step": row["step"], "practice_cases": len(groups),
            "independent_description_groups": len({text_key(g["raw_text"]) for g in groups}),
            "candidates": len(candidates), "valid_candidates": sum(bool(c["output"]["ok"]) for c in candidates),
            "informative_cases": sum(bool(g["preferences"]) for g in groups),
            "selected_independent_groups": optimizer.get("selected_independent_groups",
                len({text_key(g["raw_text"]) for g in groups if g["cid"] in selected})),
            "selected_history_groups": optimizer.get("selected_history_groups", 0),
            "experience_groups": row.get("experience_groups", 0),
            "replayed_cases": sum(g.get("provenance", {}).get("kind") in {"replay", "mixed_replay_exploration"} for g in groups),
            "exploration_candidates": sum(bool(c.get("working_skill_id")) for c in candidates),
            "exploration_cases": sum(bool(g.get("explorations")) for g in groups),
            "knowledge_preference_pairs": sum(p.get("basis") == "knowledge_tiebreak" for g in groups for p in g["preferences"]),
            "optimizer_status": optimizer.get("skipped") or ("ok" if optimizer.get("ok") else "failed"),
            "proposed_edits": len(edits) if isinstance(edits, list) else 0,
            "pre_gate_rejections": dict(Counter(t.get("reason", "unknown") for t in trials
                                               if not t.get("accept") and "consumed" not in t
                                               and t.get("stage") != "hypothesis_intake"
                                               and t.get("library_status") not in {"hypothesis", "pending"})),
            "gate_attempts": sum("consumed" in t for t in trials),
            "gates_tested": len(gates), "effective_gates": sum(g["n_affected"] >= 2 for g in gates),
            "zero_affected_gates": sum(g["n_affected"] == 0 for g in gates),
            "deferred_edits": sum(t.get("status") == "deferred" for t in trials),
            "deferred_reasons": dict(Counter(t["reason"] for t in trials if t.get("status") == "deferred")),
            "accepted_edits": sum(bool(t.get("accept")) for t in trials),
            "skill_libraries": row.get("skill_libraries", {}),
            "hypothesis_reviews": len(row.get("hypothesis_reviews", [])),
            "practice_trials": sum(len(g.get("practice_trials", [])) for g in groups),
            "practice_screened": (row.get("cultivation") or {}).get("screened", 0),
            "practice_wins": sum(t["valid_comparison"] and t["delta"] > 0 for g in groups for t in g.get("practice_trials", [])),
            "practice_losses": sum(t["valid_comparison"] and t["delta"] < 0 for g in groups for t in g.get("practice_trials", [])),
            "practice_ties": sum(t["valid_comparison"] and t["delta"] == 0 for g in groups for t in g.get("practice_trials", [])),
            "hypothesis_blockers": row.get("hypothesis_blockers", {}),
            "lifecycle_transitions": len(row.get("lifecycle_events", []))}


def diagnose_knowledge_run(run_dir):
    """Read committed artifacts only, without a model client or the current PTM data."""
    run_dir = Path(run_dir)
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if digest(protocol) != state.get("protocol_id"):
        raise ValueError("Checkpoint protocol mismatch")
    validate_checkpoint(state, protocol["partition"], require_used_gate_ids=protocol["schema"] != "knowledge-evolution-v1")
    per_step = []
    for step in range(1, state["step"] + 1):
        row = json.loads((run_dir / "steps" / f"step_{step:05d}.json").read_text(encoding="utf-8"))
        if row["step"] != step:
            raise ValueError("Committed step number mismatch")
        per_step.append(step_diagnostics(row))
    practice_size = len(protocol["partition"]["practice"])
    libraries = SkillLibraries(state.get("skill_libraries"))
    return {"steps": state["step"], "seen_cases": len(state["seen_ids"]),
            "practice_cases": practice_size, "remaining_cases": practice_size - state["cursor"],
            "rules": len(state["book"]["rules"]), "complete": state["cursor"] == practice_size,
            "experience_groups": len(state.get("experience", [])),
            **libraries.counts(SkillBook.from_state(state["book"])),
            "consumed_gate_groups": state["gate_cursor"], "deferred_edits": len(state.get("deferred", [])), "per_step": per_step}


class KnowledgeEvolver:
    def __init__(self, res, cfg, run_dir, agent, optimizer, kb=None):
        validate_config(cfg)
        self.res, self.cfg, self.kc = res, cfg, cfg.knowledge
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.agent, self.optimizer = agent, optimizer
        self.kb = kb or load_kb(res, cfg)
        self.retriever = KnowledgeRetriever(self.kb, cfg.knowledge)
        self.router = SkillRouter(self.kc, agent)
        self.support_judge = ContrastSupportJudge(self.kc, agent)
        self.libraries = SkillLibraries()
        self.policy, self.judge = KnowledgePolicy(res, cfg, self.router), KnowledgeJudge(self.kb, res)
        self.cultivate = self.policy.mode == "semantic" and self.kc.get("learning_mode", "cultivate") == "cultivate"
        self.explore = self.cultivate and bool(self.kc.get("exploration_views", ["syndrome", "integrated"]))
        self.experience = ExperiencePool(self.kc.get("experience_capacity", 256))
        data_sig = dataset_signature(res)
        archive_path = self.run_dir / "replay_archive.json"
        replay_paths = list(dict.fromkeys(str(Path(p).resolve()) for p in self.kc.get("replay_runs", [])))
        if archive_path.exists():
            self.replay_archive = json.loads(archive_path.read_text(encoding="utf-8"))
            if [s["run_dir"] for s in self.replay_archive["sources"]] != replay_paths:
                raise ValueError("Resume refused: replay sources changed")
        else:
            self.replay_archive = read_replay_sources(replay_paths, res, self.kb.version, data_sig)
        self.replay = {g["cid"]: g for g in self.replay_archive["groups"]}
        self.partition = grouped_partition(res, int(self.kc.n_practice), int(cfg.data.seed))
        exposed = self.replay_archive["exposed_ids"]
        if any(cid not in res.cases or res.cases[cid].split != "train" for cid in exposed):
            raise ValueError("Invalid replay exposure ledger")
        exposed_keys = {text_key(raw_text(res.cases[cid])) for cid in exposed}
        self.partition["replay_exposed_ids"] = exposed
        self.partition["excluded_replay_gate_groups"] = [cid for cid in self.partition["gate_pool"]
                                                        if text_key(raw_text(res.cases[cid])) in exposed_keys]
        self.partition["gate_pool"] = [cid for cid in self.partition["gate_pool"]
                                       if text_key(raw_text(res.cases[cid])) not in exposed_keys]
        if len(self.partition["gate_pool"]) < int(self.kc.gate_size) + int(self.kc.confirm_size):
            raise ValueError("Not enough fresh training groups for a gate and confirmation")
        self.book = SkillBook(int(cfg.skillbook.max_general), int(cfg.skillbook.max_scoped), 200)
        self.step, self.cursor, self.gate_cursor, self.seen_ids = 0, 0, 0, []
        self.rejected, self.deferred, self.used_gate_ids = [], [], []
        plain = to_plain(cfg)
        # Worker count and cache location may change without changing the experiment.
        protocol_cfg = copy.deepcopy(plain)
        for key in ("workers", "cache_path", "api_key_env"):
            protocol_cfg["llm"].pop(key, None)
        protocol_cfg["paths"].pop("runs_dir", None)
        self.protocol = {"schema": "knowledge-evolution-v7", "kb_version": self.kb.version,
                         "dataset_signature": data_sig, "config": protocol_cfg,
                         "replay_archive_signature": digest(self.replay_archive),
                         "implementation": {name: digest((Path(__file__).parent / name).read_text(encoding="utf-8"))
                                            for name in ("knowledge.py", "knowledge_policy.py", "knowledge_evolution.py",
                                                         "knowledge_experience.py", "knowledge_routing.py", "knowledge_lifecycle.py", "knowledge_cultivation.py", "knowledge_exploration.py",
                                                         "skillbook.py", "textindex.py", "herbs.py", "llm.py", "mock.py")},
                         "partition": self.partition, "input_mode": "raw_description",
                         "effective_agent_thinking": False, "effective_optimizer_thinking": bool(cfg.skillbook.optimizer_thinking),
                         "score_semantics": "F1 primary with optional equal-F1 knowledge dominance preferences; no gold-aware evaluation selection",
                         "gate_semantics": "label-blind preflight; frozen membership and activations before prescriptions; insufficient coverage deferred; heuristic gate",
                         "experience_semantics": "committed practice contrasts only; replay is off-policy; independent raw-description groups",
                         "library_semantics": "maturity states; executable working skills enter a subset of ordinary training solutions; evaluation uses formal skills"}
        self.protocol["practice_semantics"] = "multi-solution contrasts drive batch revisions; matched trials optional; independent gates only authorize formal promotion"
        self.protocol_id = digest(self.protocol)
        state_path = self.run_dir / "state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("protocol_id") != self.protocol_id:
                raise ValueError("Resume refused: data, knowledge, model, or experiment configuration changed; use a new run directory")
            self.restore(state)
        else:
            existing = self.run_dir / "protocol.json"
            if existing.exists() and digest(json.loads(existing.read_text(encoding="utf-8"))) != self.protocol_id:
                raise ValueError("Run directory contains a different knowledge protocol")
            atomic_json(self.run_dir / "protocol.json", self.protocol)
            atomic_json(self.run_dir / "config.json", plain)
            atomic_json(self.run_dir / "knowledge_manifest.json", self.kb.manifest)
            atomic_json(self.run_dir / "knowledge_snapshot.json", [asdict(e) for e in self.kb.evidence.values()])
            atomic_json(archive_path, self.replay_archive)
            atomic_json(state_path, self.state())

    def state(self):
        return {"protocol_id": self.protocol_id, "step": self.step, "cursor": self.cursor,
                "gate_cursor": self.gate_cursor, "seen_ids": list(self.seen_ids),
                "used_gate_ids": list(self.used_gate_ids), "deferred": copy.deepcopy(self.deferred),
                "book": self.book.state_dict(), "rejected": self.rejected, "experience": self.experience.snapshot(),
                "skill_libraries": self.libraries.snapshot()}

    def restore(self, state):
        validate_checkpoint(state, self.partition, require_used_gate_ids=True)
        self.step, self.cursor, self.gate_cursor = state["step"], state["cursor"], state["gate_cursor"]
        self.seen_ids, self.rejected = list(state["seen_ids"]), list(state["rejected"])
        self.used_gate_ids = list(state["used_gate_ids"])
        self.deferred = copy.deepcopy(state.get("deferred", []))
        groups = state.get("experience", [])
        if (not isinstance(groups, list) or len(groups) > int(self.kc.get("experience_capacity", 256))
                or len({text_key(g["raw_text"]) for g in groups}) != len(groups)
                or any(g["cid"] not in self.seen_ids or g["raw_text"] != raw_text(self.res.cases[g["cid"]])
                       or g["gold"] != list(self.res.cases[g["cid"]].herbs) for g in groups)):
            raise ValueError("Experience must contain unique committed practice description groups only")
        self.experience = ExperiencePool(self.kc.get("experience_capacity", 256), groups)
        self.book = SkillBook.from_state(state["book"])
        self.libraries = SkillLibraries(state.get("skill_libraries"))
        self.libraries.validate(self.res, self.seen_ids, self.book, self.partition["gate_pool"], int(self.kc.min_support),
                                require_formal_registry=self.policy.mode == "semantic", cultivate=self.cultivate)
        self.retriever.set_seen(self.res.get(self.seen_ids))

    def generate(self, cases, book, views=True, desc="knowledge_rx", activations=None):
        items = []
        for case in cases:
            if views:
                packets = self.retriever.packets(raw_text(case), case.cid)
            else:
                pool = self.retriever.pool(raw_text(case), case.cid)
                packets = [self.retriever.packet(pool, self.kc.eval_view)]
            items.extend((case, packet) for packet in packets)
        outputs = self.policy.run(book, [c for c, _ in items], [p for _, p in items], self.agent, desc, activations=activations)
        return [(case, packet, output) for (case, packet), output in zip(items, outputs)]

    def _optimizer_context(self, groups):
        semantic = self.policy.mode == "semantic"
        informative = (self.experience.select(groups, int(self.kc.n_reflect)) if semantic else
                       sorted((g for g in groups if g["preferences"]), key=lambda g: -g["f1_gap"]))
        payload = {"book": self.book.state_dict(), "rejected": self.rejected[-15:], "cases": [], "evidence": {},
                   "deferred": [{"text": d["edit"].get("text", ""), "reason": d["reason"], "eligibility": d["eligibility"],
                                 "required_conditions": d["edit"].get("details", {}).get("required_conditions",
                                     d["edit"].get("details", {}).get("conditions", []))} for d in self.deferred[-15:]],
                   "constraints": {"min_independent_support": int(self.kc.min_support), "max_edits": int(self.kc.k_edit),
                                   "conditions_match": "all_exact_substrings", "scope_match": "any_exact_substring_or_empty",
                                   "exclude_match": "any_exact_substring_blocks", "counterexamples_format": "case_ids_only"}}
        if semantic:
            payload["hypotheses"] = self.libraries.summaries("hypothesis")
            payload["pending_skills"] = self.libraries.summaries("pending")
            payload["constraints"].update(conditions_match="required_conditions_only; decision_steps_and_branches_are_not_prerequisites",
                                          scope_match="retrieval_only", exclude_match="known_present_blocks; unknown_is_not_absent",
                                          support_evidence="verbatim_case_quotes_required",
                                          knowledge_evidence="pure_procedural_principles_may_use_case_contrasts_only; medical_claims_require_KB_ids",
                                          skill_schema=TYPED_SKILL_SCHEMA,
                                          draft_support="hypotheses_may_have_insufficient_or_zero_support; origins_are_not_verified_support",
                                          learning_mode="cultivate" if self.cultivate else "observe",
                                          readiness="independent_training_groups; observed_advantage; optional_matched_evidence_nonworsening; independent_gate_for_formal_only"
                                                    if self.cultivate else "independent_observational_support")
        library_summaries = {name: payload[name] for name in ("hypotheses", "pending_skills") if name in payload}
        selected = []
        for group in informative[:int(self.kc.n_reflect)]:
            trial = copy.deepcopy(payload)
            trial.update(copy.deepcopy(library_summaries))
            trial.pop("omitted_library_summaries", None)
            candidates = []
            for candidate in group["candidates"]:
                packet = candidate["packet"]
                for hit in packet["evidence"]:
                    trial["evidence"][hit["id"]] = hit
                candidates.append({"view": packet["view"], "variant": packet["variant"],
                                   "practice_arm": candidate.get("practice_arm"), "hypothesis_id": candidate.get("hypothesis_id"),
                                   "working_skill_id": candidate.get("working_skill_id"), "working_maturity": candidate.get("working_maturity"),
                                   "candidate_provenance": candidate.get("candidate_provenance", "historical" if group.get("provenance", {}).get("kind") == "replay" else "current_policy"),
                                   "historical_working_skill": candidate.get("historical_working_skill"),
                                   "evidence_ids": [e["id"] for e in packet["evidence"]],
                                   "output": candidate["output"], "scores": candidate["assessment"]["scores"]})
            trial["cases"].append({"cid": group["cid"], "raw_text": group["raw_text"], "gold": group["gold"],
                                   "description_group": digest(text_key(group["raw_text"])),
                                   "experience_role": group.get("experience_role", "current"),
                                   "provenance": group.get("provenance", {"kind": "online"}),
                                   "practice_trials": group.get("practice_trials", []),
                                   "explorations": group.get("explorations", []),
                                   "preferences": group["preferences"], "candidates": candidates})
            if len(json.dumps(trial, ensure_ascii=False)) > int(self.kc.optimizer_max_chars):
                # Leave room for actual cases as the library grows; never truncate a narrative.
                important = {t["hypothesis_id"] for g in trial["cases"] for t in g.get("practice_trials", []) + g.get("explorations", [])}
                for name in ("hypotheses", "pending_skills"):
                    if name in trial:
                        trial[name].sort(key=lambda e: e["id"] not in important)
                while len(json.dumps(trial, ensure_ascii=False)) > int(self.kc.optimizer_max_chars):
                    name = next((name for name in ("hypotheses", "pending_skills")
                                 if trial.get(name) and trial[name][-1]["id"] not in important), None)
                    if name is None:
                        break
                    trial[name].pop()
                    trial.setdefault("omitted_library_summaries", 0)
                    trial["omitted_library_summaries"] += 1
                if len(json.dumps(trial, ensure_ascii=False)) > int(self.kc.optimizer_max_chars):
                    continue
            payload, selected = trial, selected + [group]
        return payload, selected

    def optimize(self, payload):
        messages = [{"role": "system", "content": OPT_SYSTEM if self.policy.mode == "semantic" else LITERAL_OPT_SYSTEM},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
        meta = {"cid": f"knowledge_opt_{self.step + 1}", "cases": payload["cases"],
                "evidence_ids": list(payload["evidence"]), "rule_texts": [r.text for r in self.book.rules],
                "k_edit": int(self.kc.k_edit), "skill_mode": self.policy.mode}
        for attempt in range(self.policy.retry_passes + 1):
            result = self.optimizer.chat_json(messages, "knowledge_opt", meta=meta,
                                              max_tokens=int(self.cfg.skillbook.optimizer_max_tokens))
            if not result.request_failed:
                return result
            if attempt < self.policy.retry_passes:
                time.sleep(self.policy.retry_wait)
        raise LLMRequestError(f"knowledge optimizer transport failure: {result.error}")

    def validate_edit(self, edit, groups, allow_hypothesis=False):
        if not isinstance(edit, dict) or not isinstance(edit.get("op"), str) or edit["op"] not in {"add", "replace", "delete"}:
            return None, "invalid_edit"
        if edit["op"] == "delete":
            return {"op": "delete", "id": str(edit.get("id", ""))}, None
        details = edit.get("details")
        if not isinstance(details, dict) or not isinstance(edit.get("text"), str):
            return None, "missing_structured_skill"
        by_id = {g["cid"]: g for g in groups}
        for field in ("supported_cases", "counterexamples"):
            values = details.get(field, [])
            if not isinstance(values, list) or any(not isinstance(v, str) or not v.strip() for v in values):
                return None, "invalid_case_evidence"
        support = strings(details.get("supported_cases"))
        counter = strings(details.get("counterexamples"))
        if any(cid not in by_id for cid in support + counter) or set(support) & set(counter):
            return None, "invalid_case_evidence"
        notes = details.get("counterexample_notes", [])
        if not isinstance(notes, list) or any(not isinstance(n, str) or not n.strip() for n in notes):
            return None, "invalid_counterexample_notes"
        if not allow_hypothesis and len({text_key(by_id[cid]["raw_text"]) for cid in support}) < int(self.kc.min_support):
            return None, "insufficient_independent_support"
        semantic = self.policy.mode == "semantic"
        evidence = {h["id"] for g in groups for c in g["candidates"] for h in c["packet"]["evidence"]
                    if not semantic or h.get("kind") != "case"}
        raw_refs = details.get("knowledge_ids", [])
        if not isinstance(raw_refs, list) or any(not isinstance(v, str) or not v.strip() for v in raw_refs):
            return None, "invalid_knowledge_evidence"
        references = strings(raw_refs)
        if not set(references) <= evidence or (not references and not allow_hypothesis and (not semantic or details.get("kind") != "principle")):
            return None, "invalid_knowledge_evidence"
        if semantic and not allow_hypothesis and not references and any(not by_id[cid].get("preferences") for cid in support):
            return None, "missing_case_contrast_evidence"
        if semantic and ("conditions" in details or "actions" in details):
            return None, "legacy_condition_fields; use_required_conditions_decision_steps_branches"
        condition_key, action_key = ("required_conditions", "decision_steps") if semantic else ("conditions", "actions")
        conditions, exclude = strings(details.get(condition_key)), strings(details.get("exclude"))
        if not semantic and any(not all(t in by_id[cid]["raw_text"] for t in conditions)
               or any(t in by_id[cid]["raw_text"] for t in exclude) for cid in support):
            return None, "conditions_do_not_match_support"
        actions = strings(details.get(action_key))
        if not actions or not isinstance(details.get("principle"), str) or not details["principle"].strip():
            return None, "missing_actions_or_principle"
        normalized = {condition_key: conditions, "exclude": exclude, action_key: actions,
                      "knowledge_ids": references, "supported_cases": support, "counterexamples": counter,
                      "counterexample_notes": strings(notes)}
        for key in ("pathogenesis_hypothesis", "principle", "uncertainty"):
            normalized[key] = str(details.get(key) or "")[:600]
        if semantic:
            origins = details.get("origin_cases", support)
            if not isinstance(origins, list) or any(not isinstance(cid, str) or cid not in by_id for cid in origins):
                return None, "invalid_origin_cases"
            if (not isinstance(details.get("kind"), str) or details["kind"] not in {"principle", "pattern"}
                    or not isinstance(details.get("applicability"), str) or not details["applicability"].strip()
                    or not strings(details.get("retrieval_terms"))):
                return None, "missing_semantic_applicability"
            for field in ("required_conditions", "decision_steps", "exclude", "retrieval_terms"):
                if not isinstance(details.get(field), list) or any(not isinstance(v, str) or not v.strip() for v in details[field]):
                    return None, "invalid_semantic_conditions"
            branches = details.get("branches")
            if not isinstance(branches, list) or any(
                not isinstance(b, dict) or not isinstance(b.get("when"), str) or not b["when"].strip()
                or b.get("kind", "patient") not in ("patient", "procedure")
                or not isinstance(b.get("actions"), list) or not b["actions"]
                or any(not isinstance(a, str) or not a.strip() for a in b["actions"]) for b in branches
            ):
                return None, "invalid_conditional_branches"
            anchors = details.get("support_evidence")
            if (not isinstance(anchors, list) or any(not isinstance(a, dict) for a in anchors)
                    or len(anchors) != len(support) or {a.get("cid") for a in anchors if isinstance(a.get("cid"), str)} != set(support)):
                return None, "invalid_support_evidence"
            for anchor in anchors:
                quotes = anchor.get("case_quotes")
                if (not isinstance(quotes, list) or not quotes
                        or any(not isinstance(q, str) or not q.strip() or q not in by_id[anchor["cid"]]["raw_text"] for q in quotes)
                        or not isinstance(anchor.get("reason"), str) or not anchor["reason"].strip()):
                    return None, "invalid_support_quotes"
            normalized.update(kind=details["kind"], applicability=details["applicability"],
                              retrieval_terms=strings(details["retrieval_terms"]), support_evidence=copy.deepcopy(anchors),
                              branches=[{"when": b["when"], "actions": strings(b["actions"]), "kind": b.get("kind", "patient")} for b in branches],
                              origin_cases=list(dict.fromkeys(origins)),
                              schema=TYPED_SKILL_SCHEMA, evidence_basis="case_contrast_and_knowledge" if references else "case_contrast")
        if len(json.dumps(normalized, ensure_ascii=False)) > 5000:
            return None, "skill_too_large"
        normalized["evidence_status"] = "source_ids_and_anchors_checked; definition_applicability_and_behavioral_gate_required; mechanism_not_independently_verified"
        return {"op": edit["op"], "id": str(edit.get("id", "")), "text": edit["text"],
                "scope": strings(edit.get("scope")), "details": normalized}, None

    def check_definition(self, proposal, applied):
        if self.policy.mode != "semantic" or applied[0]["op"] == "delete":
            return None
        rule = proposal.get(applied[0]["id"])
        review = self.router.review_definition(rule)
        if review["ok"] and not rule.details["knowledge_ids"] and review["evidence_type"] != "procedural":
            review.update(ok=False, reason="medical_claims_require_knowledge")
        return review

    def check_support(self, proposal, applied, selected):
        if self.policy.mode != "semantic" or applied[0]["op"] == "delete":
            return []
        rule = proposal.get(applied[0]["id"])
        by_id = {g["cid"]: g for g in selected}
        return [{"cid": cid, **self.router.judge(rule, by_id[cid]["raw_text"], cid)}
                for cid in rule.details["supported_cases"]]

    def _target_signature(self, edit):
        op = edit.get("op")
        rule = self.book.get(edit.get("id", "")) if isinstance(op, str) and op in {"replace", "delete"} else None
        return digest([rule.id, rule.text, rule.scope, rule.details]) if rule else None

    def _remember_hypothesis(self, edit, selected, row):
        raw = edit if isinstance(edit, dict) else {"op": "invalid", "text": "", "raw_response": edit}
        proposed = copy.deepcopy(raw)
        revises = raw.get("revises")
        parent = self.libraries.get(revises) if isinstance(revises, str) else None
        revision = raw.get("op") == "revise" or (raw.get("op") == "replace" and not raw.get("id") and bool(revises))
        if revision and parent:
            source = parent["normalized"] or parent["edit"]
            proposed["op"] = "replace" if parent["status"] == "formal" else source.get("op", "add")
            proposed["id"] = parent["formal_rule_id"] if parent["status"] == "formal" else source.get("id", "")
            if proposed["op"] not in ("add", "replace"):
                proposed["op"] = "add"
                proposed["id"] = ""
        normalized, reason = self.validate_edit(proposed, selected, allow_hypothesis=True)
        target = self._target_signature(normalized or raw)
        if normalized and normalized["op"] in {"replace", "delete"} and target is None:
            normalized, reason = None, "unknown_id"
        if (revises and parent is None) or (revision and not parent):
            normalized, reason = None, "unknown_hypothesis_revision"
        parents = raw.get("parents", [])
        if not isinstance(parents, list) or any(not isinstance(p, str) or self.libraries.get(p) is None for p in parents):
            normalized, reason, parents = None, "unknown_hypothesis_parent", []
        if parent:
            parents = list(dict.fromkeys([parent["id"], *parents]))
        if normalized and normalized["op"] != "delete":
            normalized["text"] = self.book._clean_text(normalized["text"])
            normalized["scope"] = self.book._clean_scope(normalized["scope"])
            if len(normalized["text"]) < 6:
                normalized, reason = None, "empty_text"
        entry, fresh = self.libraries.remember(raw, normalized, [g["cid"] for g in selected], self.step + 1, reason, target)
        if revises and self.libraries.get(revises) and entry["id"] != revises:
            entry["revises"] = revises
        if fresh:
            entry["parents"] = [p for p in parents if p != entry["id"]]
            entry["revision_reason"] = raw.get("revision_reason", "")
        if revision and normalized is not None:
            row.setdefault("revision_normalizations", []).append({"hypothesis_id": entry["id"], "input_op": raw["op"],
                                                                   "effective_op": normalized["op"], "parent": revises})
        row["trials"].append({"edit": raw, "stage": "hypothesis_intake", "hypothesis_id": entry["id"],
                              "library_status": entry["status"], "reason": reason or ("hypothesis_retained" if fresh else "hypothesis_merged"),
                              "accept": False})

    @staticmethod
    def _hypothesis_rule(entry):
        edit = entry["normalized"]
        return Rule(id=entry["id"], text=edit["text"], scope=edit["scope"], details=edit["details"])

    def _hypothesis_cases(self, entry, groups):
        unique = {}
        for group in [*self.experience.groups.values(), *groups]:
            key = group_key(group["raw_text"])
            if group["preferences"] and key not in entry["observations"]:
                unique.setdefault(key, group)
        candidates = list(unique.values())
        if not candidates:
            return []
        rule = self._hypothesis_rule(entry)
        index = KnowledgeBase.make_index([(g["cid"], g["raw_text"]) for g in candidates])
        scores = dict(index.query([routing_text(rule)], len(candidates))[0])
        claimed = set(entry["normalized"]["details"]["supported_cases"])
        origins = set(entry["origin_cases"])
        candidates = [g for g in candidates if g["cid"] in origins or scores.get(g["cid"], 0) > 0 or rule.details["kind"] == "principle"]
        return sorted(candidates, key=lambda g: (g["cid"] not in claimed, g["cid"] not in origins,
                                                -scores.get(g["cid"], 0), g["cid"]))

    def _promotion_edit(self, entry):
        edit = copy.deepcopy(entry["normalized"])
        if edit["op"] == "delete":
            return edit
        if self.cultivate:
            witnesses = [t for t in entry["practice_observations"].values() if t["valid_comparison"]][:int(self.kc.min_support)]
            edit["details"]["supported_cases"] = []
            edit["details"]["support_evidence"] = []
            edit["details"]["practice_cases"] = [t["cid"] for t in witnesses]
            edit["details"]["practice_summary"] = practice_summary(entry)
            exploration = list(entry.get("exploration_observations", {}).values())
            edit["details"]["exploration_cases"] = [o["cid"] for o in exploration if any(r["valid"] for r in o["outcomes"])][:int(self.kc.min_support)]
            edit["details"]["exploration_summary"] = exploration_summary(entry)
            edit["details"]["evidence_status"] = "observed_multi_solution_contrasts_and_optional_matched_trials; not_causal; independent_gate_required"
            return edit
        # Keep the full evidence in the library; the formal rule holds a bounded witness set.
        support = positive_observations(entry)[:int(self.kc.min_support)]
        edit["details"]["supported_cases"] = [o["cid"] for o in support]
        edit["details"]["support_evidence"] = [{"cid": o["cid"], "case_quotes": [q[:200] for q in o["contrast"]["case_quotes"][:2]],
                                                "reason": o["contrast"]["reason"][:600]} for o in support]
        edit["details"]["origin_cases"] = entry["origin_cases"][:int(self.kc.n_reflect)]
        edit["details"]["evidence_status"] = "applicability_and_observed_contrasts_checked; mechanism_not_proven"
        return edit

    def _advance_libraries(self, groups, row):
        step = self.step + 1
        row["hypothesis_reviews"] = []
        for entry in self.libraries.entries:
            if entry["status"] == "formal" or entry["needs_revision"] or entry.get("archived") or entry["normalized"] is None:
                continue
            edit = entry["normalized"]
            if edit["op"] == "delete":
                self.libraries.move(entry, "pending", "deletion_awaiting_gate", step)
                continue
            if entry["definition_review"] is None:
                review = self.router.review_definition(self._hypothesis_rule(entry))
                if review["ok"] and not edit["details"]["knowledge_ids"] and (
                    edit["details"]["kind"] != "principle" or review["evidence_type"] != "procedural"
                ):
                    review.update(ok=False, reason="medical_claims_require_knowledge")
                entry["definition_review"] = review
                if not review["ok"]:
                    entry["needs_revision"] = True
                    self.libraries.move(entry, "hypothesis", review["reason"], step)
                    row["trials"].append({"edit": edit, "hypothesis_id": entry["id"], "stage": "definition_validation",
                                          "library_status": "hypothesis", "reason": review["reason"], "accept": False,
                                          "definition_review": review})
                    continue
                previous = self.libraries.get(entry.get("revises"))
                if previous and previous["status"] != "formal" and not self.cultivate:
                    previous["archived"] = True
                    self.libraries.move(previous, "hypothesis", "superseded_by_revision", step)
        eligible = [e for e in self.libraries.entries if e["status"] == "hypothesis" and not e["needs_revision"]
                    and not e.get("archived") and e["normalized"] is not None and e["normalized"]["op"] != "delete"]
        eligible.sort(key=lambda e: (e["last_review_step"], e["created_step"], e["id"]))
        queues = {} if self.cultivate else {e["id"]: self._hypothesis_cases(e, groups) for e in eligible}
        budget = 0 if self.cultivate else int(self.kc.get("hypothesis_review_budget", 12))
        while budget > 0:
            progressed = False
            for entry in eligible:
                if budget == 0:
                    break
                queue = queues[entry["id"]]
                if not queue or len(positive_observations(entry)) >= int(self.kc.min_support):
                    continue
                group = queue.pop(0)
                rule = self._hypothesis_rule(entry)
                applicability = self.router.judge(rule, group["raw_text"], group["cid"])
                contrast = self.support_judge.assess(rule, group) if applicability["decision"] == "applicable" else None
                observation = {"cid": group["cid"], "case_signature": digest(group["raw_text"]), "step": step,
                               "episode_step": group.get("observed_step"), "episode_signature": digest(group),
                               "preferences": copy.deepcopy(group["preferences"]), "applicability": applicability}
                if contrast is not None:
                    observation["contrast"] = contrast
                entry["observations"][group_key(group["raw_text"])] = observation
                entry["last_review_step"] = step
                row["hypothesis_reviews"].append({"hypothesis_id": entry["id"], **observation})
                budget -= 1
                progressed = True
            if not progressed:
                break
        for entry in eligible:
            enough = (cultivation_ready(entry, int(self.kc.min_support)) if self.cultivate
                      else len(positive_observations(entry)) >= int(self.kc.min_support))
            self.libraries.move(entry, "pending" if enough else "hypothesis",
                                "awaiting_behavioral_gate" if enough else
                                ("learning_from_solution_contrasts" if self.cultivate else "awaiting_independent_contrast_support"), step)
        pending = sorted((e for e in self.libraries.entries if e["status"] == "pending" and not e.get("archived")),
                         key=lambda e: (e["last_gate_step"], e["created_step"], e["id"]))
        for entry in pending[:int(self.kc.get("pending_gate_budget", 1))]:
            if (entry["needs_revision"] or entry["gate_failed"] or entry["normalized"] is None
                    or (entry["normalized"]["op"] != "delete" and (not (entry["definition_review"] or {}).get("ok")
                        or (not cultivation_ready(entry, int(self.kc.min_support)) if self.cultivate
                            else len(positive_observations(entry)) < int(self.kc.min_support))))):
                self.libraries.move(entry, "hypothesis", "pending_requirements_not_met", step)
                continue
            edit = self._promotion_edit(entry)
            if edit["op"] in {"replace", "delete"} and self._target_signature(edit) != entry["target_signature"]:
                entry["needs_revision"] = True
                self.libraries.move(entry, "hypothesis", "stale_formal_target", step)
                continue
            if len(json.dumps(edit.get("details", {}), ensure_ascii=False)) > 5000:
                entry["needs_revision"] = True
                self.libraries.move(entry, "hypothesis", "formal_evidence_too_large", step)
                continue
            proposal, applied, skipped = self.book.apply([edit], step)
            if not applied:
                entry["needs_revision"] = True
                self.libraries.move(entry, "hypothesis", skipped[0][1], step)
                continue
            entry["last_gate_step"] = step
            decision = self.gate(proposal, scan_state=entry["gate_scan"])
            entry["gate_scan"] = decision.get("scan_state")
            entry["gate_history"].append({"step": step, "reason": decision["reason"], "plan_path": decision.get("plan_path"),
                                           "consumed": decision["consumed"], "eligibility": decision["eligibility"],
                                           "delta": decision.get("gate", {}).get("delta")})
            if decision["accept"]:
                self.book = proposal
                rid = applied[0]["id"]
                self.libraries.retire(rid, step, "removed_or_replaced_in_formal_book", except_id=entry["id"])
                if edit["op"] == "delete":
                    entry["archived"] = True
                    self.libraries.move(entry, "hypothesis", "completed_deletion", step)
                else:
                    self.book.get(rid).gain = (decision["confirmation"] or decision["gate"])["delta"]
                    entry["formal_rule_id"] = rid
                    self.libraries.move(entry, "formal", "behavioral_gate_accepted", step)
                    if self.cultivate:
                        for pid in entry.get("parents", []):
                            previous = self.libraries.get(pid)
                            if previous["status"] != "formal":
                                previous["archived"] = True
                                self.libraries.move(previous, "hypothesis", "superseded_by_validated_revision", step)
            elif decision["status"] == "deferred":
                self.libraries.move(entry, "pending", decision["reason"], step)
            else:
                entry.update(needs_revision=True, gate_failed=True)
                self.libraries.move(entry, "hypothesis", "behavioral_gate_rejected; revision_required", step)
            row["trials"].append({"edit": edit, "hypothesis_id": entry["id"], "library_status": entry["status"],
                                  "definition_review": entry["definition_review"],
                                  "support_checks": [o["applicability"] for o in positive_observations(entry)], **decision})
        self.deferred = [{"step": e["last_gate_step"], "edit": e["normalized"], "reason": e["reason"],
                          "eligibility": e["gate_history"][-1]["eligibility"]} for e in self.libraries.entries
                         if e["status"] == "pending" and e["gate_history"] and e["gate_history"][-1]["reason"].startswith("insufficient")]

    def _compare(self, base, proposal, ids, label, activations=None):
        cases = self.res.get(ids)
        old = self.generate(cases, base, views=False, desc=label + "_old",
                            activations=activations["before"] if activations else None)
        new = self.generate(cases, proposal, views=False, desc=label + "_new",
                            activations=activations["after"] if activations else None)
        rows = []
        for (case, packet, before), (_, _, after) in zip(old, new):
            scores_old = self.judge.evaluate(raw_text(case), before, case.herbs)["scores"]
            scores_new = self.judge.evaluate(raw_text(case), after, case.herbs)["scores"]
            rows.append({"cid": case.cid, "packet": packet, "before": before, "after": after,
                         "scores_before": scores_old, "scores_after": scores_new})
        def prompt(book, row, arm):
            packet = row["packet"]
            if activations:
                packet = {"_packet": packet, "_activation": activations[arm][row["cid"]]}
            return self.policy.prompt_hash(book, self.res.cases[row["cid"]], packet)
        changed = [r for r in rows if prompt(base, r, "before") != prompt(proposal, r, "after")]
        delta, t = paired_stats([r["scores_after"]["f1"] for r in changed], [r["scores_before"]["f1"] for r in changed])
        guards = {key: sum(r["scores_after"][key] - r["scores_before"][key] for r in rows)
                  for key in ("risk_flags", "citation_errors", "anchor_errors", "oov_count")}
        return {"n": len(rows), "n_affected": len(changed), "delta": delta,
                "t": t if math.isfinite(t) else 99.0, "guard_deltas": guards, "rows": rows}

    def plan_gate(self, proposal, scan_state=None):
        """Select by actual prompt activation, never by a reference prescription or score."""
        count = int(self.kc.gate_size) + int(self.kc.confirm_size)
        used = set(self.used_gate_ids)
        available = [cid for cid in self.partition["gate_pool"] if cid not in used]
        if self.policy.mode == "semantic":
            return self._semantic_gate_plan(proposal, available, count, scan_state)
        affected = [cid for cid in available
                    if self.policy.rule_context(self.book, self.res.cases[cid])[0]
                    != self.policy.rule_context(proposal, self.res.cases[cid])[0]]
        eligibility = {"fresh_groups": len(available), "affected_groups": len(affected), "required_groups": count,
                       "selection": "prompt change from raw narrative only; no gold or scores"}
        if len(affected) < count:
            reason = "no_fresh_affected_groups" if not affected else "insufficient_fresh_affected_groups"
            return {"ready": False, "reason": reason, "ids": [], "eligibility": eligibility}
        return {"ready": True, "ids": affected[:count], "eligibility": eligibility}

    def _semantic_gate_plan(self, proposal, available, count, scan_state=None):
        context_key = digest([self.book.signature(), [(r.id, r.scope, routing_skill(r), r.details.get("knowledge_ids"),
                                                      r.details.get("uncertainty"), r.details.get("evidence_basis")) for r in proposal.rules]])
        previous = scan_state if scan_state and scan_state.get("context_key") == context_key else {}
        examined_before = set(previous.get("examined_ids", []))
        before = {r.id: r for r in self.book.rules}
        after = {r.id: r for r in proposal.rules}
        changed = [rule for rid in sorted(before.keys() | after.keys())
                   if rid not in before or rid not in after or routing_skill(before[rid]) != routing_skill(after[rid])
                   or before[rid].details != after[rid].details or before[rid].scope != after[rid].scope
                   for rule in (before.get(rid), after.get(rid)) if rule is not None]
        scores = {}
        if available and changed:
            index = KnowledgeBase.make_index([(cid, raw_text(self.res.cases[cid])) for cid in available])
            for hits in index.query([routing_text(r) for r in changed], len(available)):
                for cid, score in hits:
                    scores[cid] = max(score, scores.get(cid, 0.0))
        general = any(r.details.get("kind") == "principle" for r in changed)
        order = {cid: i for i, cid in enumerate(available)}
        candidates = sorted((cid for cid in available if cid not in examined_before and (scores.get(cid, 0.0) > 0 or general)),
                            key=lambda cid: (-scores.get(cid, 0.0), order[cid]))[:int(self.kc.gate_candidate_limit)]
        affected, examined, routes = [], [], {"before": {}, "after": {}}
        # Previously eligible groups have never been scored or allocated. Revalidate their routes
        # under the same operative definitions, then continue past the previously examined prefix.
        for cid in previous.get("affected_ids", []):
            if cid not in order:
                continue
            case = self.res.cases[cid]
            old, new = self.router.resolve(self.book, case), self.router.resolve(proposal, case)
            if self.policy.rule_context(self.book, case, old)[0] != self.policy.rule_context(proposal, case, new)[0]:
                affected.append(cid)
                routes["before"][cid], routes["after"][cid] = old, new
            if len(affected) == count:
                break
        for cid in candidates:
            if len(affected) == count:
                break
            case = self.res.cases[cid]
            old, new = self.router.resolve(self.book, case), self.router.resolve(proposal, case)
            examined.append({"cid": cid, "before": old, "after": new})
            if self.policy.rule_context(self.book, case, old)[0] != self.policy.rule_context(proposal, case, new)[0]:
                affected.append(cid)
                routes["before"][cid], routes["after"][cid] = old, new
            if len(affected) == count:
                break
        examined_set = examined_before | {e["cid"] for e in examined}
        examined_ids = [cid for cid in available if cid in examined_set]
        next_scan = {"context_key": context_key, "examined_ids": examined_ids, "affected_ids": affected}
        eligibility = {"fresh_groups": len(available), "candidate_groups": len(candidates), "examined_groups": len(examined),
                       "cumulative_examined_groups": len(examined_ids), "resumed_scan": bool(previous),
                       "affected_groups": len(affected), "required_groups": count, "candidate_limit": int(self.kc.gate_candidate_limit),
                       "selection": "bounded narrative-only retrieval and semantic applicability; no gold, prescriptions or scores",
                       "coverage_is_global": len(examined_ids) == len(available)}
        return {"ready": len(affected) >= count, "ids": affected if len(affected) >= count else [],
                "reason": None if len(affected) >= count else "insufficient_semantic_coverage_within_budget",
                "eligibility": eligibility, "activations": routes, "examined": examined, "scan_state": next_scan}

    def frozen_gate_plan(self, proposal, scan_state=None):
        identity = {"protocol_id": self.protocol_id, "step": self.step + 1, "base": self.book.signature(),
                    "proposal": proposal.signature(), "seen_ids": self.seen_ids, "used_gate_ids": self.used_gate_ids}
        if scan_state is not None:
            identity["scan_signature"] = digest(scan_state)
        name = digest(identity) + ".json"
        path = self.run_dir / "gate_plans" / name
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["identity"] != identity or record["plan_signature"] != digest(record["plan"]):
                raise ValueError("Frozen gate plan changed")
            plan = record["plan"]
        else:
            plan = self.plan_gate(proposal, scan_state)
            # Commit membership and both routing decisions before any gate prescription or score.
            atomic_json(path, {"identity": copy.deepcopy(identity), "plan": plan, "plan_signature": digest(plan)})
        return plan, str(Path("gate_plans") / name)

    def gate(self, proposal, scan_state=None):
        plan, plan_path = self.frozen_gate_plan(proposal, scan_state)
        if not plan["ready"]:
            return {"accept": False, "status": "deferred", "reason": plan["reason"], "consumed": [],
                    "eligibility": plan["eligibility"], "plan_path": plan_path, "scan_state": plan.get("scan_state")}
        ids = plan["ids"]
        self.used_gate_ids.extend(ids)
        self.gate_cursor = len(self.used_gate_ids)
        gate_ids, confirm_ids = ids[:int(self.kc.gate_size)], ids[int(self.kc.gate_size):]
        kwargs = {"activations": plan["activations"]} if "activations" in plan else {}
        result = self._compare(self.book, proposal, gate_ids, "fresh_gate", **kwargs)
        ok = (result["n_affected"] >= 2 and result["delta"] >= float(self.kc.min_delta)
              and result["t"] >= float(self.kc.t_min) and all(d <= 0 for d in result["guard_deltas"].values()))
        confirmation = None
        if ok and confirm_ids:
            confirmation = self._compare(self.book, proposal, confirm_ids, "fresh_confirm", **kwargs)
            ok = (confirmation["n_affected"] >= 2 and confirmation["delta"] >= 0
                  and all(d <= 0 for d in confirmation["guard_deltas"].values()))
        return {"accept": bool(ok), "status": "accepted" if ok else "rejected",
                "reason": "accepted" if ok else "behavioral_gate_rejected", "eligibility": plan["eligibility"],
                "consumed": ids, "gate": result, "confirmation": confirmation, "plan_path": plan_path,
                "scan_state": plan.get("scan_state")}

    def run_step(self):
        self.libraries.events = []
        ids = self.partition["practice"][self.cursor:self.cursor + int(self.kc.batch)]
        groups = {cid: {"cid": cid, "raw_text": raw_text(self.res.cases[cid]), "gold": list(self.res.cases[cid].herbs),
                        "candidates": [], "provenance": {"kind": "online", "book_signature": self.book.signature(),
                                                          "protocol_id": self.protocol_id}} for cid in ids}
        fresh = []
        for cid in ids:
            episode = self.replay.get(cid)
            if episode is not None and replay_compatible(episode, self.seen_ids, self.res):
                groups[cid] = copy.deepcopy(episode)
            elif self.kc.get("replay_only"):
                raise ValueError(f"No compatible replay for practice case {cid}; preserve source batch/order or allow fresh generation")
            else:
                fresh.append(cid)
        # All candidates are generated before any target is passed to the judge or optimizer.
        exploration = None
        if self.explore and not self.kc.get("collect_only"):
            exploration = generate_exploration(self, list(groups.values()), fresh, atomic_json)
        else:
            generated = self.generate(self.res.get(fresh), self.book) if fresh else []
            for case, packet, output in generated:
                groups[case.cid]["candidates"].append({"packet": packet, "output": output})
        cultivation = (generate_practice(self, list(groups.values()), atomic_json)
                       if self.cultivate and not self.kc.get("collect_only") and int(self.kc.get("hypothesis_trial_budget", 0)) > 0 else None)
        for group in groups.values():
            group["observed_step"] = self.step + 1
            candidates = group["candidates"]
            for candidate in candidates:
                candidate["assessment"] = self.judge.evaluate(group["raw_text"], candidate["output"], group["gold"])
            scores = [c["assessment"]["scores"]["f1"] for c in candidates]
            group["f1_gap"] = max(scores) - min(scores)
            group["preferences"] = preference_pairs(candidates, float(self.kc.min_gap),
                knowledge_tiebreak=self.policy.mode == "semantic" and bool(self.kc.get("knowledge_tiebreak", True)),
                knowledge_min_gap=float(self.kc.get("knowledge_min_gap", 0.05)))
            group["unique_prescriptions"] = len({tuple(sorted(c["output"]["herbs"])) for c in candidates})
            group["unique_evidence_sets"] = len({tuple(sorted(e["id"] for e in c["packet"]["evidence"])) for c in candidates})
            group["oracle_f1"] = max(scores)  # diagnostic upper bound, never a deployable prediction
        if cultivation is not None:
            record_practice(self, list(groups.values()))
        if exploration is not None:
            record_exploration(self, list(groups.values()))
        row = {"step": self.step + 1, "seen_before": list(self.seen_ids), "groups": list(groups.values()), "trials": []}
        row["cultivation"] = cultivation
        row["exploration"] = exploration
        payload, selected = self._optimizer_context(row["groups"])
        n_independent = len({text_key(g["raw_text"]) for g in selected})
        selection = {"selected_cases": [g["cid"] for g in selected], "selected_independent_groups": n_independent,
                     "selected_history_groups": sum(g.get("experience_role") == "history" for g in selected)}
        minimum = 1 if self.policy.mode == "semantic" else int(self.kc.min_support)
        if not self.kc.get("collect_only") and n_independent >= minimum:
            result = self.optimize(payload)
            row["optimizer"] = {"ok": result.ok, "error": result.error, "data": result.data,
                                **selection}
            data = result.data if result.ok and isinstance(result.data, dict) else {}
            edits = data.get("edits") if isinstance(data.get("edits"), list) else []
            for edit in edits[:int(self.kc.k_edit)]:
                if self.policy.mode == "semantic":
                    self._remember_hypothesis(edit, selected, row)
                    continue
                normalized, reason = self.validate_edit(edit, selected)
                if reason:
                    row["trials"].append({"edit": edit, "stage": "validation", "reason": reason, "accept": False})
                    self.rejected.append({"text": edit.get("text", "") if isinstance(edit, dict) else "",
                                          "stage": "validation", "reason": reason})
                    continue
                proposal, applied, skipped = self.book.apply([normalized], self.step + 1)
                if not applied:
                    row["trials"].append({"edit": normalized, "stage": "application", "accept": False, "reason": skipped[0][1]})
                    self.rejected.append({"text": normalized.get("text", ""), "stage": "application", "reason": skipped[0][1]})
                    continue
                definition_review = self.check_definition(proposal, applied)
                if definition_review is not None and not definition_review["ok"]:
                    row["trials"].append({"edit": normalized, "stage": "definition_validation", "reason": definition_review["reason"],
                                          "accept": False, "definition_review": definition_review})
                    self.rejected.append({"text": normalized["text"], "stage": "definition_validation", "reason": definition_review["reason"],
                                          "definition_review": definition_review})
                    continue
                support_checks = self.check_support(proposal, applied, selected)
                if any(check["decision"] != "applicable" for check in support_checks):
                    reason = "support_not_semantically_applicable"
                    row["trials"].append({"edit": normalized, "stage": "support_validation", "reason": reason,
                                          "accept": False, "support_checks": support_checks, "definition_review": definition_review})
                    self.rejected.append({"text": normalized["text"], "stage": "support_validation", "reason": reason})
                    continue
                decision = self.gate(proposal)
                row["trials"].append({"edit": normalized, "definition_review": definition_review, "support_checks": support_checks, **decision})
                if decision["accept"]:
                    self.book = proposal
                    changed = self.book.get(applied[0]["id"])
                    if changed is not None:
                        changed.gain = (decision["confirmation"] or decision["gate"])["delta"]
                elif decision.get("status") == "deferred":
                    self.deferred.append({"step": self.step + 1, "edit": normalized, "reason": decision["reason"],
                                          "eligibility": decision["eligibility"]})
                else:
                    self.rejected.append({"text": normalized.get("text", ""), "reason": decision["reason"],
                                          "delta": decision.get("gate", {}).get("delta")})
        else:
            row["optimizer"] = {"skipped": "collect_only" if self.kc.get("collect_only") else "insufficient_independent_informative_cases",
                                **selection}
        if self.policy.mode == "semantic" and not self.kc.get("collect_only"):
            self._advance_libraries(row["groups"], row)
        self.experience.add(row["groups"])
        self.seen_ids += [cid for cid in ids if cid not in self.seen_ids]
        self.cursor += len(ids)
        self.step += 1
        self.retriever.set_seen(self.res.get(self.seen_ids))
        row["book_after"] = self.book.state_dict()
        row["experience_groups"] = len(self.experience.groups)
        row["skill_libraries"] = self.libraries.counts(self.book)
        row["hypothesis_blockers"] = dict(Counter(e["reason"] for e in self.libraries.entries
                                                  if e["needs_revision"] and not e.get("archived")))
        row["lifecycle_events"] = copy.deepcopy(self.libraries.events)
        row["diagnostics"] = step_diagnostics(row)
        return row

    def train(self, max_steps=None):
        while self.cursor < len(self.partition["practice"]) and (max_steps is None or self.step < max_steps):
            before = copy.deepcopy(self.state())
            try:
                row = self.run_step()
                atomic_json(self.run_dir / "steps" / f"step_{self.step:05d}.json", row)
                atomic_json(self.run_dir / "state.json", self.state())
            except Exception:
                self.restore(before)
                raise
        atomic_json(self.run_dir / "skillbook_final.json", self.book.state_dict())
        (self.run_dir / "skillbook_final.md").write_text(self.book.to_markdown(), encoding="utf-8")
        atomic_json(self.run_dir / "hypothesis_library.json", [e for e in self.libraries.entries if e["status"] == "hypothesis"])
        atomic_json(self.run_dir / "pending_library.json", [e for e in self.libraries.entries if e["status"] == "pending"])
        last = json.loads((self.run_dir / "steps" / f"step_{self.step:05d}.json").read_text(encoding="utf-8")) if self.step else None
        return {"steps": self.step, "seen_cases": len(self.seen_ids), "rules": len(self.book.rules),
                "complete": self.cursor == len(self.partition["practice"]),
                "experience_groups": len(self.experience.groups),
                **self.libraries.counts(self.book),
                "consumed_gate_groups": self.gate_cursor, "deferred_edits": len(self.deferred),
                "last_step_diagnostics": step_diagnostics(last) if last else None}


def evaluate_knowledge(res, cfg, kb, state, agent, cases, empty=False):
    retriever = KnowledgeRetriever(kb, cfg.knowledge)
    retriever.set_seen(res.get(state["seen_ids"]))
    book = SkillBook() if empty else SkillBook.from_state(state["book"])
    policy, judge = KnowledgePolicy(res, cfg), KnowledgeJudge(kb, res)
    packets = [retriever.packet(retriever.pool(raw_text(c), c.cid), cfg.knowledge.eval_view) for c in cases]
    outputs = policy.run(book, cases, packets, agent, "knowledge_eval")
    rows = [{"cid": c.cid, "raw_text": raw_text(c), "packet": packet, "output": output,
             "assessment": judge.evaluate(raw_text(c), output, c.herbs)} for c, packet, output in zip(cases, packets, outputs)]
    summary = {"n": len(rows), "rules": len(book.rules), "selection": "one fixed knowledge view; no target-aware selection",
               "view": cfg.knowledge.eval_view, "memory_cases": len(state["seen_ids"])}
    if rows:
        for metric in rows[0]["assessment"]["scores"]:
            values = [row["assessment"]["scores"][metric] for row in rows if row["assessment"]["scores"][metric] is not None]
            summary[metric] = float(np.mean(values)) if values else None
    return rows, summary
