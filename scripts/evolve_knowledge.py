"""Knowledge inspection, multi-view skill evolution, and frozen single-output evaluation."""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import _bootstrap  # noqa: F401

from tcmskill.config import apply_override, load_config, to_cfg
from tcmskill.data import Resources
from tcmskill.knowledge import KnowledgeRetriever, digest, raw_text
from tcmskill.knowledge_evolution import (
    KnowledgeEvolver, atomic_json, dataset_signature, diagnose_knowledge_run, evaluate_knowledge, load_kb, validate_checkpoint, validate_config,
)
from tcmskill.llm import build_llm


def make_llm(cfg, directory, optimizer=False):
    cfg = copy.deepcopy(cfg)
    cfg.llm.thinking = bool(cfg.skillbook.optimizer_thinking) if optimizer else False
    if cfg.llm.provider == "mock":
        cfg.llm.cache_path = ""
    llm = build_llm(cfg)
    llm.call_log_path = str(directory / ("optimizer_calls.jsonl" if optimizer else "agent_calls.jsonl"))
    return llm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "train", "eval", "diagnose"])
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--llm", choices=["mock", "deepseek", "openai_compatible"])
    parser.add_argument("--run_name", default="knowledge_v7")
    parser.add_argument("--run_dir", help="Training run directory; eval loads its saved configuration and memory")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--reuse_run", action="append", default=[], help="Replay committed practice candidates from a historical run; repeatable")
    parser.add_argument("--replay_only", action="store_true", help="Require historical episodes; exploration slots, optional matched trials and gates still use the current agent")
    parser.add_argument("--collect_only", action="store_true", help="Store contrast experience without draft trials, optimizer or gate calls")
    parser.add_argument("--text", help="Original narrative for inspection; never a prescription")
    parser.add_argument("--cid", help="Inspect one processed case using desc_raw, without its gold herbs")
    parser.add_argument("--out")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--empty", action="store_true", help="Evaluate without skills but with identical revealed memory and retrieval")
    args = parser.parse_args()
    if args.command != "train" and (args.reuse_run or args.replay_only or args.collect_only):
        parser.error("Replay/collection flags are only supported for train")
    if args.command == "diagnose":
        if not args.run_dir:
            parser.error("diagnose requires --run_dir")
        print(json.dumps(diagnose_knowledge_run(args.run_dir), ensure_ascii=False, indent=2))
        return
    if args.command == "eval":
        if not args.run_dir or not args.out:
            parser.error("eval requires --run_dir and --out")
        run_dir = Path(args.run_dir)
        cfg = to_cfg(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
        for override in args.set:
            if not override.startswith("llm."):
                parser.error("eval freezes training data/retrieval settings; only llm.* overrides are supported")
            apply_override(cfg, override)
    else:
        cfg = load_config(args.config, overrides=args.set)
        run_dir = Path(args.run_dir or Path(cfg.paths.runs_dir) / args.run_name)
    cfg.data.input_mode = "raw_description"
    if args.reuse_run:
        cfg.knowledge.replay_runs = args.reuse_run
    if args.replay_only:
        cfg.knowledge.replay_only = True
    if args.collect_only:
        cfg.knowledge.collect_only = True
    if args.llm:
        cfg.llm.provider = args.llm
    validate_config(cfg)
    res = Resources(cfg)
    kb = load_kb(res, cfg)
    if args.command == "inspect":
        if bool(args.text) == bool(args.cid) or not args.out:
            parser.error("inspect requires exactly one of --text/--cid and an --out directory")
        text, cid = (raw_text(res.cases[args.cid]), args.cid) if args.cid else (args.text, "inspection")
        retriever = KnowledgeRetriever(kb, cfg.knowledge)
        out = Path(args.out)
        if (out / "inspection.json").exists():
            parser.error("Inspection output already exists; choose a new --out")
        atomic_json(out / "knowledge_manifest.json", kb.manifest)
        atomic_json(out / "knowledge_snapshot.json", [asdict(e) for e in kb.evidence.values()])
        packets = retriever.packets(text, cid)
        atomic_json(out / "inspection.json", {"pool": retriever.pool(text, cid), "packets": packets})
        print(json.dumps({"kb_version": kb.version, "counts": kb.manifest["counts"],
                          "packets": len(packets), "out": str(out)}, ensure_ascii=False))
        return
    if args.command == "train":
        run_dir.mkdir(parents=True, exist_ok=True)
        agent, optimizer = make_llm(cfg, run_dir), make_llm(cfg, run_dir, True)
        experiment = KnowledgeEvolver(res, cfg, run_dir, agent, optimizer, kb)
        try:
            summary = experiment.train(args.max_steps)
        finally:
            atomic_json(run_dir / "usage_current_session.json", {"agent": agent.usage.summary(), "optimizer": optimizer.usage.summary()})
        print(json.dumps(summary, ensure_ascii=False))
        return
    state = json.loads((run_dir / "state.json").read_text(encoding="utf-8"))
    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    if digest(protocol) != state.get("protocol_id") or protocol["kb_version"] != kb.version or protocol["dataset_signature"] != dataset_signature(res):
        raise ValueError("Evaluation refused: training protocol, knowledge or dataset changed")
    validate_checkpoint(state, protocol["partition"], require_used_gate_ids=protocol["schema"] != "knowledge-evolution-v1")
    source_dir = Path(__file__).resolve().parents[1] / "src" / "tcmskill"
    if any(digest((source_dir / name).read_text(encoding="utf-8")) != signature for name, signature in protocol["implementation"].items()):
        raise ValueError("Evaluation refused: implementation changed since training")
    if args.n <= 0:
        parser.error("--n must be positive")
    out = Path(args.out)
    if (out / "predictions.json").exists():
        parser.error("Evaluation output already exists; choose a new --out")
    out.mkdir(parents=True, exist_ok=True)
    agent = make_llm(cfg, out)
    rows, summary = evaluate_knowledge(res, cfg, kb, state, agent, res.get(res.select(args.split, args.n)), args.empty)
    atomic_json(out / "predictions.json", rows)
    atomic_json(out / "metrics.json", summary)
    atomic_json(out / "protocol.json", {"training_protocol_id": state["protocol_id"], "llm": dict(cfg.llm),
                                        "effective_agent_thinking": False,
                                        "split": args.split, "empty": args.empty, "state_signature": digest(state)})
    atomic_json(out / "usage_current_session.json", agent.usage.summary())
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
