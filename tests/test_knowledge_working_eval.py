"""Working evaluation must freeze evidence, pair contexts and never train on test labels."""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmskill.config import load_config  # noqa: E402
from tcmskill.data import Case  # noqa: E402
from tcmskill.knowledge import Evidence, KnowledgeBase, digest  # noqa: E402
from tcmskill.knowledge_cultivation import draft_context, working_entries  # noqa: E402
from tcmskill.knowledge_evolution import atomic_json, dataset_signature  # noqa: E402
from tcmskill.knowledge_lifecycle import SkillLibraries  # noqa: E402
from tcmskill.knowledge_progress import TrainingProgress  # noqa: E402
from tcmskill.knowledge_routing import SkillRouter, TYPED_SKILL_SCHEMA  # noqa: E402
from tcmskill.knowledge_working_eval import FrozenWorkingContext, WorkingSkillEvaluation, paired_summary, read_json  # noqa: E402
from tcmskill.llm import LLMRequestError, build_llm  # noqa: E402
from tcmskill.skillbook import Rule, SkillBook  # noqa: E402


class Lexicon:
    labels = ["herb-a", "herb-b"]

    def normalize_list(self, names):
        return list(dict.fromkeys(n for n in names if n in self.labels)), [n for n in names if n not in self.labels]


def add_entry(libraries, text="Compare evidence for cough before adding herbs"):
    edit = {"op": "add", "text": text, "scope": ["cough"], "details": {
        "schema": TYPED_SKILL_SCHEMA, "kind": "principle", "retrieval_terms": ["cough"],
        "applicability": "cough", "required_conditions": ["cough"], "decision_steps": ["Check evidence"],
        "branches": [], "exclude": [], "principle": "Compare evidence", "pathogenesis_hypothesis": "uncertain",
        "knowledge_ids": [], "evidence_basis": "case_contrast", "uncertainty": "not validated"}}
    entry, _ = libraries.remember(edit, edit, ["tr0"], 1)
    entry["definition_review"] = {"ok": True}
    return entry


@pytest.fixture
def fixture(tmp_path):
    cfg = load_config(str(ROOT / "configs/default.yaml"), overrides=["llm.provider=mock", "llm.cache_path=",
        "skillbook.retry_wait=0", "skillbook.retry_passes=0", "knowledge.skill_mode=semantic", "knowledge.min_support=3"])
    kb = KnowledgeBase([Evidence("K1", "relation", "herb-a for cough", herbs=["herb-a"], terms=["cough"])], {"fixture": 1})
    cases = [Case("tr0", "train", "cleaned", [], ["herb-a"], desc_raw="cough in a training case"),
             Case("tr1", "train", "cleaned", [], ["herb-a"], desc_raw="cough in an unseen gate case"),
             Case("te0", "test", "cleaned", [], ["herb-b"], desc_raw="cough in the first test narrative"),
             Case("te1", "test", "cleaned", [], ["herb-b"], desc_raw="cough in the second test narrative")]
    res = SimpleNamespace(cases={c.cid: c for c in cases}, lexicon=Lexicon(), labels=Lexicon.labels,
                          herb_freq={h: 1 for h in Lexicon.labels},
                          incompat=SimpleNamespace(violations=lambda herbs: []),
                          herb_info=SimpleNamespace(same_category=lambda a, b: False))
    res.get = lambda ids: [res.cases[cid] for cid in ids]
    libraries = SkillLibraries()
    add_entry(libraries)
    protocol = {"kb_version": kb.version, "dataset_signature": dataset_signature(res),
                "implementation": {"knowledge_evolution.py": digest((ROOT / "src/tcmskill/knowledge_evolution.py").read_text())},
                "partition": {"practice": ["tr0"], "gate_pool": ["tr1"]}}
    state = {"protocol_id": digest(protocol), "cursor": 1, "gate_cursor": 0, "seen_ids": ["tr0"],
             "used_gate_ids": [], "book": SkillBook().state_dict(), "skill_libraries": libraries.snapshot()}
    return SimpleNamespace(cfg=cfg, res=res, kb=kb, state=state, protocol=protocol,
                           cases=cases[2:], out=tmp_path / "evaluation")


def evaluator(f, **kwargs):
    args = dict(res=f.res, cfg=f.cfg, kb=f.kb, state=f.state, training_protocol=f.protocol,
                cases=f.cases, out=f.out, implementation={"fixture": "1"})
    args.update(kwargs)
    return WorkingSkillEvaluation(**args)


def improved_agent(f, monkeypatch):
    agent = build_llm(f.cfg)
    original = agent._h_knowledge_rx

    def answer(meta):
        result = original(meta)
        result["herbs"][0]["name"] = "herb-b" if meta["rule_ids"] else "herb-a"
        return result

    monkeypatch.setattr(agent, "_h_knowledge_rx", answer)
    return agent


def test_paired_evaluation_is_blind_immutable_and_saves_before_scoring(fixture, monkeypatch):
    f = fixture
    before = digest(f.state)
    evaluation, agent = evaluator(f), improved_agent(f, monkeypatch)
    original_call, original_score = agent.chat_json, evaluation.judge.evaluate
    requests = []

    def call(messages, task, meta=None, **kwargs):
        assert task in {"knowledge_rx", "knowledge_applicability"}
        assert "gold" not in meta and "herb-b" not in str(meta)
        if task == "knowledge_rx":
            assert f.res.cases[meta["cid"]].desc_raw in messages[1]["content"]
            assert "cleaned" not in messages[1]["content"]
            path = f.out / "cases" / digest(meta["cid"]) / "plan.json"
            assert path.exists()
            requests.append(meta)
        return original_call(messages, task, meta=meta, **kwargs)

    def score(text, output, gold):
        directory = f.out / "cases" / digest(output["cid"])
        assert (directory / "baseline.json").exists() and (directory / "working.json").exists()
        return original_score(text, output, gold)

    monkeypatch.setattr(agent, "chat_json", call)
    monkeypatch.setattr(evaluation.judge, "evaluate", score)
    result = evaluation.run(agent)
    assert result["complete"] and result["n"] == result["activation_n"] == 2
    assert result["all"]["delta_f1"] == result["activated"]["delta_f1"] == 1
    assert result["formal_skills"] == 0 and result["eligible_working_skills"] == 1
    assert len(requests) == 4 and requests[0]["packet"] == requests[1]["packet"]
    assert {e["cid"] for e in requests[0]["packet"]["evidence"] if e["kind"] == "case"} == {"tr0"}
    assert digest(f.state) == before and digest(evaluation.context.libraries.snapshot()) == digest(f.state["skill_libraries"])
    assert result["all"]["working"]["known_n"]["f1"] == 2


def test_first_applicable_retrieved_skill_not_highest_scoring_candidate(fixture, monkeypatch):
    f = fixture
    libraries = SkillLibraries(f.state["skill_libraries"])
    add_entry(libraries, "Second cough evidence strategy")
    add_entry(libraries, "Third cough evidence strategy")
    f.state["skill_libraries"] = libraries.snapshot()
    evaluation, agent = evaluator(f), build_llm(f.cfg)
    original = agent._h_knowledge_applicability
    checked = []

    def applicability(meta):
        checked.append(meta["skill"]["id"])
        result = original(meta)
        if meta["skill"]["id"] == "H1":
            result["requirements"][1].update(status="unknown", quotes=[])
        return result

    monkeypatch.setattr(agent, "_h_knowledge_applicability", applicability)
    monkeypatch.setattr(SkillRouter, "retrieve", lambda self, book, text:
                        [{"id": r.id, "score": 1.0} for r in book.rules])
    result = evaluation.run(agent)
    assert result["selected_skill_counts"] == {"H2": 2}
    assert checked == ["H1", "H2", "H1", "H2"]


@pytest.mark.parametrize("mode", ["unknown", "empty_library", "bad_definition"])
def test_no_applicable_skill_reuses_baseline_instead_of_sampling_advantage(fixture, monkeypatch, mode):
    f = fixture
    if mode == "empty_library":
        f.state["skill_libraries"] = SkillLibraries().snapshot()
    if mode == "bad_definition":
        f.state["skill_libraries"]["entries"][0]["definition_review"] = {"ok": False}
    agent = build_llm(f.cfg)
    original = agent._h_knowledge_applicability

    def unknown(meta):
        result = original(meta)
        result["requirements"][1].update(status="unknown", quotes=[])
        return result

    monkeypatch.setattr(agent, "_h_knowledge_applicability", unknown)
    evaluation = evaluator(f)
    calls, original_run = [], evaluation.policy.run

    def run(*args, **kwargs):
        calls.append(1)
        return original_run(*args, **kwargs)

    monkeypatch.setattr(evaluation.policy, "run", run)
    result = evaluation.run(agent)
    rows = read_json(f.out / "predictions.json")
    assert len(calls) == 2 and result["activation_n"] == 0 and result["all"]["delta_f1"] == 0
    assert result["activated"]["delta_f1"] is None
    assert all(r["working"]["reused_baseline"] for r in rows)


def test_gate_failed_but_executable_skill_is_reported_not_removed(fixture):
    f = fixture
    f.state["skill_libraries"]["entries"][0].update(gate_failed=True, needs_revision=True)
    result = evaluator(f).run(build_llm(f.cfg))
    assert result["eligible_gate_failed"] == 1 and result["gate_failed_activated"]["n"] == 2


def test_activated_metrics_compare_the_same_subset_and_keep_full_test_denominator(fixture, monkeypatch):
    agent = improved_agent(fixture, monkeypatch)
    original = agent._h_knowledge_applicability

    def applicability(meta):
        result = original(meta)
        if meta["cid"] == "te1":
            result["requirements"][1].update(status="unknown", quotes=[])
        return result

    monkeypatch.setattr(agent, "_h_knowledge_applicability", applicability)
    result = evaluator(fixture).run(agent)
    assert result["all"]["n"] == 2 and result["all"]["delta_f1"] == 0.5
    assert result["activated"]["n"] == 1 and result["activated"]["delta_f1"] == 1
    assert result["activation_rate"] == 0.5


def test_unknown_metric_is_not_imputed_as_zero():
    rows = [{arm: {"assessment": {"scores": {"f1": 0.5, "knowledge": value}}}
             for arm in ("baseline", "working")} for value in (None, 0.8)]
    result = paired_summary(rows)
    assert result["baseline"]["knowledge"] == 0.8 and result["baseline"]["known_n"]["knowledge"] == 1


def test_version_filter_and_formal_replacement_use_existing_semantics(fixture):
    libraries = SkillLibraries(fixture.state["skill_libraries"])
    child = add_entry(libraries, "Newer cough strategy")
    child["parents"] = ["H1"]
    context = FrozenWorkingContext(SkillBook(), libraries)
    assert [e["id"] for e in working_entries(context)] == ["H2"]
    child["gate_failed"] = True
    assert [e["id"] for e in working_entries(context)] == ["H1", "H2"]
    context.book.rules = [Rule("R1", "original cough strategy"), Rule("R2", "other strategy")]
    child["normalized"].update(op="replace", id="R1")
    child["target_signature"] = context._target_signature(child["normalized"])
    context.router = SkillRouter(fixture.cfg.knowledge, build_llm(fixture.cfg))
    case = fixture.cases[0]
    before = context.router.resolve(context.book, case)
    book, route, _ = draft_context(context, child, case, before)
    assert {r.id for r in book.rules} == {"R2", "H2"} and "R1" not in route["applied_ids"]
    context.book.rules[0].text = "changed formal definition"
    assert "H2" not in [e["id"] for e in working_entries(context)]


def test_interruption_keeps_baseline_and_resume_does_not_reroute_or_regenerate_it(fixture, monkeypatch):
    f = fixture
    evaluation, agent = evaluator(f), build_llm(f.cfg)
    original = agent._h_knowledge_rx
    calls = []

    def fail_treatment(meta):
        calls.append((meta["cid"], bool(meta["rule_ids"])))
        if meta["rule_ids"]:
            raise LLMRequestError("test interruption")
        return original(meta)

    monkeypatch.setattr(agent, "_h_knowledge_rx", fail_treatment)
    with pytest.raises(LLMRequestError):
        evaluation.run(agent)
    directory = f.out / "cases" / digest("te0")
    assert (directory / "baseline.json").exists() and not (directory / "working.json").exists()
    assert not (f.out / "metrics.json").exists()
    monkeypatch.setattr(agent, "_h_knowledge_rx", original)
    resumed = evaluator(f)
    assert resumed.run(agent)["complete"]
    assert calls == [("te0", False), ("te0", True)]
    monkeypatch.setattr(agent, "chat_json", lambda *a, **kw: pytest.fail("Finished results must not issue new requests"))
    stream = io.StringIO()
    with TrainingProgress(stream=stream, label="Evaluation") as progress:
        assert evaluator(f).run(agent, progress)["complete"]
    assert "2/2" in stream.getvalue()


def test_invalid_response_is_scored_zero_not_filtered_or_retried(fixture, monkeypatch):
    agent = build_llm(fixture.cfg)
    original = agent._h_knowledge_rx

    def invalid(meta):
        result = original(meta)
        result["herbs"] = []
        return result

    monkeypatch.setattr(agent, "_h_knowledge_rx", invalid)
    summary = evaluator(fixture).run(agent)
    assert summary["all"]["working"]["f1"] == 0 and summary["all"]["working"]["valid_output"] == 0
    assert summary["all"]["working"]["known_n"]["f1"] == 2


@pytest.mark.parametrize("artifact", ["plan.json", "baseline.json", "working.json", "result.json"])
def test_tampered_case_artifacts_refuse_resume(fixture, artifact):
    evaluation = evaluator(fixture)
    evaluation.run(build_llm(fixture.cfg))
    path = fixture.out / "cases" / digest("te0") / artifact
    data = read_json(path)
    data["payload"]["changed"] = True
    atomic_json(path, data)
    with pytest.raises(ValueError, match="artifact changed"):
        evaluator(fixture).run(build_llm(fixture.cfg))


@pytest.mark.parametrize("change", ["model", "case_order", "implementation", "snapshot"])
def test_changed_experiment_cannot_mix_results_in_same_output(fixture, change):
    evaluation = evaluator(fixture)
    evaluation.prepare()
    options = {}
    if change == "model":
        fixture.cfg.llm.model = "another-model"
    elif change == "case_order":
        options["cases"] = list(reversed(fixture.cases))
    elif change == "implementation":
        options["implementation"] = {"fixture": "2"}
    else:
        fixture.state["skill_libraries"]["entries"][0]["definition_review"] = {"ok": False}
    with pytest.raises(ValueError, match="Resume refused"):
        evaluator(fixture, **options).prepare()


def test_training_hash_mismatch_and_train_case_input_are_refused(fixture):
    fixture.protocol["implementation"]["knowledge_evolution.py"] = "modified"
    fixture.state["protocol_id"] = digest(fixture.protocol)
    with pytest.raises(ValueError, match="implementation changed"):
        evaluator(fixture)
    with pytest.raises(ValueError, match="held-out"):
        evaluator(fixture, cases=[fixture.res.cases["tr0"]])


def test_planning_never_accesses_gold(fixture):
    evaluation = evaluator(fixture)
    evaluation.context.router = SkillRouter(fixture.cfg.knowledge, build_llm(fixture.cfg))

    class NoGold:
        cid = "te0"
        desc_raw = fixture.cases[0].desc_raw

        @property
        def herbs(self):
            pytest.fail("Planning accessed the held-out prescription")

    before = evaluation._plan(NoGold())
    fixture.res.cases["te0"].herbs = ["changed-test-gold"]
    assert evaluation._plan(NoGold()) == before


def test_missing_prediction_of_completed_case_is_not_silently_regenerated(fixture):
    evaluator(fixture).run(build_llm(fixture.cfg))
    (fixture.out / "cases" / digest("te0") / "baseline.json").unlink()
    with pytest.raises(ValueError, match="missing its frozen plan"):
        evaluator(fixture).run(build_llm(fixture.cfg))
