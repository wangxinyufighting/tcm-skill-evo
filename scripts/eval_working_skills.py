"""Compare a frozen formal baseline with the same baseline plus an executable working skill."""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
from uuid import uuid4

import _bootstrap  # noqa: F401

from tcmskill.config import to_cfg
from tcmskill.data import Resources
from tcmskill.knowledge import digest
from tcmskill.knowledge_evolution import atomic_json, load_kb, validate_config
from tcmskill.knowledge_progress import ProgressLLM, TrainingProgress
from tcmskill.knowledge_working_eval import WorkingSkillEvaluation, read_json
from tcmskill.llm import build_llm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out", required=True, help="Separate output directory; rerun the identical command to resume")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--llm", choices=["mock", "deepseek", "openai_compatible"])
    parser.add_argument("--prepare_only", action="store_true", help="Validate and freeze the experiment without LLM calls")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    run, out = Path(args.run_dir).resolve(), Path(args.out).resolve()
    if out == run or out in run.parents:
        parser.error("--out must not be the training directory or its ancestor")
    cfg = to_cfg(read_json(run / "config.json"))
    training_protocol, state = read_json(run / "protocol.json"), read_json(run / "state.json")
    recorded = copy.deepcopy(dict(cfg))
    recorded["llm"] = {k: v for k, v in cfg.llm.items() if k not in {"workers", "cache_path", "api_key_env"}}
    recorded["paths"] = {k: v for k, v in cfg.paths.items() if k != "runs_dir"}
    if recorded != training_protocol["config"]:
        raise ValueError("Saved configuration differs from the training protocol")
    cfg.data.input_mode = "raw_description"
    if args.llm:
        cfg.llm.provider = args.llm
    validate_config(cfg)
    res = Resources(cfg)
    ids = res.select(args.split, args.n)
    if len(ids) != args.n:
        parser.error("--n exceeds the number of held-out cases")
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), root / "scripts/_bootstrap.py"]
    files.extend(sorted((root / "src/tcmskill").glob("*.py")))
    implementation = {str(p.relative_to(root)): digest(p.read_text(encoding="utf-8")) for p in files}
    evaluation = WorkingSkillEvaluation(res, cfg, load_kb(res, cfg), state, training_protocol,
                                       res.get(ids), out, implementation)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another evaluation is using this --out directory")
        prepared = evaluation.prepare()
        if args.prepare_only:
            print(json.dumps(prepared, ensure_ascii=False))
            return
        agent_cfg = copy.deepcopy(cfg)
        agent_cfg.llm.thinking = False
        if agent_cfg.llm.provider == "mock":
            agent_cfg.llm.cache_path = ""
        agent = build_llm(agent_cfg)
        agent.call_log_path = str(out / "agent_calls.jsonl")
        try:
            with TrainingProgress(enabled=not args.no_progress, label="评测") as progress:
                result = evaluation.run(ProgressLLM(agent, progress), progress)
        finally:
            atomic_json(out / "usage_sessions" / (uuid4().hex + ".json"), agent.usage.summary())
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
