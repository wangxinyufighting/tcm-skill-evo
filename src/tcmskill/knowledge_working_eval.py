"""Frozen, paired evaluation of executable working skills, without promotion or learning."""
from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .knowledge import KnowledgeRetriever, digest, raw_text
from .knowledge_cultivation import draft_context, working_entries
from .knowledge_evolution import KnowledgeEvolver, atomic_json, dataset_signature, validate_checkpoint
from .knowledge_lifecycle import SkillLibraries, group_key
from .knowledge_policy import KnowledgeJudge, KnowledgePolicy
from .knowledge_routing import SkillRouter
from .skillbook import SkillBook


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_training(res, cfg, kb, state, protocol):
    if (digest(protocol) != state.get("protocol_id") or protocol["kb_version"] != kb.version
            or protocol["dataset_signature"] != dataset_signature(res)):
        raise ValueError("Evaluation refused: training protocol, knowledge or dataset changed")
    validate_checkpoint(state, protocol["partition"], require_used_gate_ids=True)
    source = Path(__file__).parent
    if any(digest((source / name).read_text(encoding="utf-8")) != signature
           for name, signature in protocol["implementation"].items()):
        raise ValueError("Evaluation refused: implementation changed since training")
    if cfg.knowledge.get("skill_mode") != "semantic":
        raise ValueError("Working-skill evaluation requires a semantic skill library")
    book, libraries = SkillBook.from_state(state["book"]), SkillLibraries(state["skill_libraries"])
    libraries.validate(res, state["seen_ids"], book, protocol["partition"]["gate_pool"],
                       int(cfg.knowledge.min_support), require_formal_registry=True,
                       cultivate=cfg.knowledge.get("learning_mode", "cultivate") == "cultivate")
    return book, libraries


class FrozenWorkingContext:
    """Reuse training's version/eligibility rules without constructing or mutating a trainer."""

    _target_signature = KnowledgeEvolver._target_signature
    _hypothesis_rule = staticmethod(KnowledgeEvolver._hypothesis_rule)

    def __init__(self, book, libraries):
        self.book, self.libraries, self.router = book, libraries, None


def paired_summary(rows):
    result = {"n": len(rows)}
    for arm in ("baseline", "working"):
        scores = [r[arm]["assessment"]["scores"] for r in rows]
        keys = sorted({key for score in scores for key in score})
        known = {key: [s[key] for s in scores if s.get(key) is not None] for key in keys}
        result[arm] = {key: sum(values) / len(values) if values else None for key, values in known.items()}
        result[arm]["known_n"] = {key: len(values) for key, values in known.items()}
    deltas = [r["working"]["assessment"]["scores"]["f1"] - r["baseline"]["assessment"]["scores"]["f1"] for r in rows]
    result.update(delta_f1=sum(deltas) / len(deltas) if deltas else None,
                  wins=sum(d > 1e-12 for d in deltas), losses=sum(d < -1e-12 for d in deltas),
                  ties=sum(abs(d) <= 1e-12 for d in deltas))
    return result


class WorkingSkillEvaluation:
    def __init__(self, res, cfg, kb, state, training_protocol, cases, out, implementation):
        self.res, self.cfg, self.kb = res, cfg, kb
        self.cases, self.out = list(cases), Path(out)
        if not self.cases or len({c.cid for c in self.cases}) != len(self.cases):
            raise ValueError("Evaluation requires a nonempty, unique case list")
        if any(c.split not in {"val", "test"} for c in self.cases):
            raise ValueError("Working-skill evaluation only accepts held-out cases")
        book, libraries = verify_training(res, cfg, kb, state, training_protocol)
        seen = res.get(state["seen_ids"])
        if (any(c.split != "train" for c in seen)
                or {group_key(raw_text(c)) for c in seen} & {group_key(raw_text(c)) for c in self.cases}):
            raise ValueError("Evaluation cases overlap the frozen training memory")
        self.context = FrozenWorkingContext(book, libraries)
        self.entries = working_entries(self.context, include_pending=True)
        self.by_id = {e["id"]: e for e in self.entries}
        self.work_book = SkillBook()
        self.work_book.rules = [self.context._hypothesis_rule(e) for e in self.entries]
        self.retriever = KnowledgeRetriever(kb, cfg.knowledge)
        self.retriever.set_seen(seen)
        self.policy, self.judge = KnowledgePolicy(res, cfg), KnowledgeJudge(kb, res)
        self.snapshot = {"book": state["book"], "skill_libraries": libraries.snapshot(), "seen_ids": state["seen_ids"]}
        self.counts = {"library_entries": len(libraries.entries), "formal_skills": len(book.rules),
                       "eligible_working_skills": len(self.entries),
                       "eligible_by_maturity": dict(Counter(e["status"] for e in self.entries)),
                       "eligible_gate_failed": sum(bool(e["gate_failed"]) for e in self.entries)}
        self.protocol = {"schema": "knowledge-working-eval-v1", "training_protocol_id": state["protocol_id"],
                         "state_signature": digest(state), "snapshot_signature": digest(self.snapshot),
                         "kb_version": kb.version, "dataset_signature": training_protocol["dataset_signature"],
                         "cases": [{"cid": c.cid, "split": c.split, "raw_signature": digest(raw_text(c))} for c in self.cases],
                         "llm": dict(cfg.llm), "effective_agent_thinking": False,
                         "knowledge": dict(cfg.knowledge), "prescribe": dict(cfg.prescribe),
                         "skillbook": dict(cfg.skillbook), "implementation": implementation,
                         "selection": "first applicable among relevance-ranked retrieved skills; no gold access",
                         "baseline": "frozen formal book (no skills if empty)",
                         "working": "same formal book plus one applicable working add/replace edit",
                         "no_match": "reuse baseline output when prompts are identical",
                         "gate_failed_policy": "retain executable entries as in training; report separately",
                         "view": cfg.knowledge.eval_view, "memory_cases": len(seen), **self.counts}
        self.protocol_id = digest(self.protocol)

    def prepare(self):
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / "protocol.json"
        if path.exists():
            if read_json(path) != self.protocol:
                raise ValueError("Resume refused: checkpoint, cases, model, settings or evaluator changed; use a new --out")
        else:
            if any(p.name != ".eval.lock" for p in self.out.iterdir()):
                raise ValueError("Output directory is not an empty working-skill evaluation")
            atomic_json(path, self.protocol)
        snapshot = self.out / "snapshot.json"
        if snapshot.exists() and read_json(snapshot) != self.snapshot:
            raise ValueError("Frozen evaluation snapshot changed")
        if not snapshot.exists():
            atomic_json(snapshot, self.snapshot)
        return {"prepared": True, "n": len(self.cases), **self.counts, "out": str(self.out)}

    def _load(self, path):
        if not path.exists():
            return None
        saved = read_json(path)
        if saved.get("protocol_id") != self.protocol_id or saved.get("signature") != digest(saved["payload"]):
            raise ValueError(f"Frozen evaluation artifact changed: {path}")
        return saved["payload"]

    def _save(self, path, payload):
        atomic_json(path, {"protocol_id": self.protocol_id, "signature": digest(payload), "payload": payload})

    def _plan(self, case):
        text = raw_text(case)
        packet = self.retriever.packet(self.retriever.pool(text, case.cid), self.cfg.knowledge.eval_view)
        before = self.context.router.resolve(self.context.book, case)
        book, after, selected, screened = self.context.book, before, None, []
        for hit in self.context.router.retrieve(self.work_book, text):
            entry = self.by_id[hit["id"]]
            proposal, route, decision = draft_context(self.context, entry, case, before)
            screened.append({**hit, "decision": decision})
            if proposal is not None:
                book, after = proposal, route
                selected = {"id": entry["id"], "definition_key": entry["definition_key"],
                            "maturity": entry["status"], "gate_failed": bool(entry["gate_failed"])}
                break
        plan = {"cid": case.cid, "raw_signature": digest(text), "packet": packet,
                "selected": selected, "screened": screened, "working_book": book.state_dict(),
                "baseline_route": before, "working_route": after}
        for arm, current, route in (("baseline", self.context.book, before), ("working", book, after)):
            plan[arm + "_prompt"] = self.policy.prompt_hash(current, case, {"_packet": packet, "_activation": route})
        return plan

    def _case(self, case, agent):
        # Only this label-free object reaches retrieval, routing and generation.
        blind = SimpleNamespace(cid=case.cid, desc_raw=raw_text(case))
        directory = self.out / "cases" / digest(case.cid)
        plan = self._load(directory / "plan.json")
        if plan is None:
            plan = self._plan(blind)
            self._save(directory / "plan.json", plan)
        if plan["cid"] != case.cid or plan["raw_signature"] != digest(raw_text(case)):
            raise ValueError("Frozen evaluation case changed")
        outputs = {}
        for arm, book in (("baseline", self.context.book), ("working", SkillBook.from_state(plan["working_book"]))):
            route = plan[arm + "_route"]
            prompt = self.policy.prompt_hash(book, blind, {"_packet": plan["packet"], "_activation": route})
            if prompt != plan[arm + "_prompt"]:
                raise ValueError("Frozen evaluation prompt changed")
            saved = self._load(directory / (arm + ".json"))
            if saved is None:
                reused = arm == "working" and prompt == plan["baseline_prompt"]
                output = (copy.deepcopy(outputs["baseline"]["output"]) if reused else
                          self.policy.run(book, [blind], [plan["packet"]], agent, "working_eval_" + arm,
                                          activations={case.cid: route})[0])
                saved = {"plan_signature": digest(plan), "prompt": prompt, "output": output, "reused_baseline": reused}
                self._save(directory / (arm + ".json"), saved)
            if saved["plan_signature"] != digest(plan) or saved["prompt"] != prompt:
                raise ValueError("Saved output does not match the frozen plan")
            outputs[arm] = saved
        # Both predictions are durable before opening the reference for scoring.
        row = {"cid": case.cid, "raw_text": raw_text(case), "selected": plan["selected"],
               "activated": plan["working_prompt"] != plan["baseline_prompt"],
               "plan": str((directory / "plan.json").relative_to(self.out))}
        for arm, saved in outputs.items():
            row[arm] = {"output": saved["output"], "reused_baseline": saved["reused_baseline"],
                        "assessment": self.judge.evaluate(raw_text(case), saved["output"], case.herbs)}
        self._save(directory / "result.json", row)
        return row

    def summarize(self, rows):
        activated = [r for r in rows if r["activated"]]
        failed_gate = [r for r in activated if r["selected"]["gate_failed"]]
        return {"protocol_id": self.protocol_id, "n": len(rows), "target_n": len(self.cases),
                "complete": len(rows) == len(self.cases), **self.counts,
                "memory_cases": self.protocol["memory_cases"], "view": self.protocol["view"],
                "activation_n": len(activated), "activation_rate": len(activated) / len(rows) if rows else None,
                "selected_skill_counts": dict(Counter(r["selected"]["id"] for r in activated)),
                "all": paired_summary(rows), "activated": paired_summary(activated),
                "gate_failed_activated": paired_summary(failed_gate),
                "interpretation": "exploratory frozen working-library evaluation, not formal promotion or clinical validation"}

    def run(self, agent, progress=None):
        self.prepare()
        self.context.router = SkillRouter(self.cfg.knowledge, agent)
        self.policy.router = self.context.router
        rows = []
        for case in self.cases:
            directory = self.out / "cases" / digest(case.cid)
            if self._load(directory / "result.json") is None:
                break
            if any(self._load(directory / name) is None for name in ("plan.json", "baseline.json", "working.json")):
                raise ValueError("Completed case is missing its frozen plan or predictions")
            rows.append(self._case(case, agent))
        if progress:
            progress.configure(len(self.cases), len(rows), len(rows), 1)
        for case in self.cases[len(rows):]:
            if progress:
                progress.begin_step(len(rows), len(rows) + 1)
            rows.append(self._case(case, agent))
            atomic_json(self.out / "metrics.json", self.summarize(rows))
        summary = self.summarize(rows)
        atomic_json(self.out / "metrics.json", summary)
        atomic_json(self.out / "predictions.json", rows)
        if progress:
            progress.finish(len(rows), len(rows), True)
        return summary
