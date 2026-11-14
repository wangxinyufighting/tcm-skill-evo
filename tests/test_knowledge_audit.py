"""Regressions for the condition/type and broad-scope errors in the real v7 pilot."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tcmskill.config import load_config  # noqa: E402
from tcmskill.knowledge_evolution import KnowledgeEvolver  # noqa: E402
from tcmskill.knowledge_routing import SkillRouter  # noqa: E402
from tcmskill.llm import LLMRequestError, LLMResult, build_llm  # noqa: E402
from tcmskill.skillbook import Rule, SkillBook  # noqa: E402

RAW = "胸膈痞满，腹胁胀痛，短气噎闷，咳呕痰水，噫醋吞酸，哕逆恶心。"
SCOPE = "成人与小儿的胃系病症；不用于无胃系症状或纯表证者。"


@pytest.fixture
def router():
    cfg = load_config(str(ROOT / "configs/default.yaml"), overrides=["llm.provider=mock", "llm.cache_path="])
    return SkillRouter(cfg.knowledge, build_llm(cfg))


def rule(when="饮食减少、四肢倦懈突出", kind="patient"):
    return Rule("H1", "ACTION_ONLY_MARKER", details={
        "kind": "pattern", "applicability": SCOPE, "required_conditions": ["胃系症状至少一项"],
        "exclude": ["纯表证"], "knowledge_ids": [],
        "branches": [{"when": when, "kind": kind, "actions": ["ACTION_ONLY_MARKER"]}],
        "supported_cases": ["HIDDEN_CASE"], "support_evidence": [{"reason": "HIDDEN_EVIDENCE"}]})


def disputed_review(router, monkeypatch, role="mixed"):
    original = router.llm._h_knowledge_skill_review

    def review(meta):
        data = original(meta)
        data["evidence_type"] = "mixed"
        data["branch_roles"][0].update(role=role, reason="患者条件后有选药动作，或比较涉及医学知识。")
        return data

    monkeypatch.setattr(router.llm, "_h_knowledge_skill_review", review)


def branch_role(condition, role):
    return {"index": condition["index"], "role": role, "reason": "只按条件的判断对象分类。",
            "patient_quotes": [condition["when"]] if role == "patient" else [],
            "procedure_quotes": [condition["when"]] if role == "procedure" else []}


@pytest.mark.parametrize("when,kind", [
    ("饮食减少、四肢倦懈突出", "patient"),
    ("梦遗失精与盗汗并见", "patient"),
    ("候选方加入较多健脾渗湿或和胃降逆药", "procedure"),
    ("候选方以黄芪、党参、白术、茯苓等补益健脾为主而缺少白附子、羌活等风痰主药", "procedure"),
])
def test_pilot_role_errors_rechecked_without_actions_or_medical_axis(router, monkeypatch, when, kind):
    disputed_review(router, monkeypatch)
    calls, original = [], router.llm.chat_json

    def capture(messages, task, **kwargs):
        if task == "knowledge_branch_conditions":
            payload = json.loads(messages[1]["content"])
            assert payload == {"conditions": [{"index": 0, "when": when}]}
            assert set(kwargs["meta"]) == {"cid", "conditions"}
            assert "ACTION_ONLY_MARKER" not in messages[1]["content"]
            calls.append(payload)
        return original(messages, task, **kwargs)

    monkeypatch.setattr(router.llm, "chat_json", capture)
    monkeypatch.setattr(router.llm, "_h_knowledge_branch_conditions",
                        lambda m: {"branch_roles": [branch_role(m["conditions"][0], kind)]})
    skill = rule(when, kind)
    result = router.review_definition(skill)
    assert result["ok"] and result["evidence_type"] == "mixed"
    assert result["reported_branch_roles"][0]["role"] == "mixed"
    assert result["branch_roles"][0]["role"] == kind and len(calls) == 1
    result["branch_roles"][0]["role"] = "tampered"
    assert router.review_definition(skill)["branch_roles"][0]["role"] == kind and len(calls) == 1


@pytest.mark.parametrize("actual", ["patient", "mixed", "unclear"])
def test_patient_facts_hidden_in_procedure_are_still_blocked(router, monkeypatch, actual):
    disputed_review(router, monkeypatch)
    skill = rule("患者四肢倦懈且候选方缺少益气药", "procedure")
    response = branch_role({"index": 0, "when": skill.details["branches"][0]["when"]}, actual)
    if actual == "mixed":
        response.update(patient_quotes=["患者四肢倦懈"], procedure_quotes=["候选方缺少益气药"])
    monkeypatch.setattr(router.llm, "_h_knowledge_branch_conditions", lambda m: {"branch_roles": [response]})
    result = router.review_definition(skill)
    assert not result["ok"] and result["reason"] == "misplaced_applicability_conditions"
    assert result["branch_condition_recheck"]["ok"]


@pytest.mark.parametrize("bad", ["index", "missing", "action_quote", "mixed_overlap", "wrong_axis"])
def test_branch_recheck_must_be_complete_and_anchored_to_when(router, monkeypatch, bad):
    disputed_review(router, monkeypatch)
    response = branch_role({"index": 0, "when": "饮食减少、四肢倦懈突出"}, "patient")
    if bad == "index":
        response["index"] = True
    elif bad == "action_quote":
        response["patient_quotes"] = ["ACTION_ONLY_MARKER"]
    elif bad == "mixed_overlap":
        response.update(role="mixed", procedure_quotes=["四肢倦懈"])
    elif bad == "wrong_axis":
        response["role"] = "medical"
    monkeypatch.setattr(router.llm, "_h_knowledge_branch_conditions",
                        lambda m: {"branch_roles": [] if bad == "missing" else [response]})
    result = router.review_definition(rule())
    assert not result["ok"] and result["reason"] == "invalid_definition_review"
    assert not result["branch_condition_recheck"]["ok"]


def test_uncontested_definition_does_not_add_a_recheck_call(router, monkeypatch):
    monkeypatch.setattr(router.llm, "_h_knowledge_branch_conditions", lambda m: pytest.fail("Unneeded branch review"))
    assert router.review_definition(rule())["ok"]


def test_role_correction_does_not_waive_medical_knowledge_requirement(router, monkeypatch):
    disputed_review(router, monkeypatch)
    monkeypatch.setattr(router.llm, "_h_knowledge_branch_conditions",
                        lambda m: {"branch_roles": [branch_role(m["conditions"][0], "patient")]})
    exp = KnowledgeEvolver.__new__(KnowledgeEvolver)
    exp.router, exp.policy = router, SimpleNamespace(mode="semantic")
    book = SkillBook()
    book.rules = [rule()]
    result = exp.check_definition(book, [{"op": "add", "id": "H1"}])
    assert not result["ok"] and result["reason"] == "medical_claims_require_knowledge"


def unknown_scope(router, monkeypatch):
    original = router.llm._h_knowledge_applicability

    def judge(meta):
        data = original(meta)
        data["decision"] = "uncertain"
        data["requirements"][0].update(status="unknown", quotes=[], reason="未说明具体年龄，未明确排除纯表证。")
        return data

    monkeypatch.setattr(router.llm, "_h_knowledge_applicability", judge)


def scope_parts():
    return [
        {"text": "成人与小儿的", "kind": "unrestricted_scope", "status": "not_required", "quotes": ["成人与小儿"],
         "reason": "该措辞覆盖年龄组，不要求患者的具体年龄。"},
        {"text": "胃系病症；", "kind": "patient_condition", "status": "supported", "quotes": ["咳呕痰水", "哕逆恶心"],
         "reason": "呕恶等原文表现支持胃系病症。"},
        {"text": "不用于无胃系症状或纯表证者。", "kind": "exclusion", "status": "unknown", "quotes": [],
         "reason": "并未明确排除纯表证，保留未知，不据此补造否定事实。"},
    ]


def test_pilot_broad_scope_keeps_patient_facts_and_unknown_exclusions(router, monkeypatch):
    unknown_scope(router, monkeypatch)
    calls, original = [], router.llm.chat_json

    def capture(messages, task, **kwargs):
        if task == "knowledge_scope_review":
            payload = json.loads(messages[1]["content"])
            assert payload["raw_text"] == RAW and payload["applicability"] == SCOPE
            assert payload["required_conditions"] == ["胃系症状至少一项"] and payload["exclusions"] == ["纯表证"]
            assert len(payload["verified_requirements"]) == 1 and payload["verified_requirements"][0]["status"] == "supported"
            assert set(payload) == {"raw_text", "applicability", "required_conditions", "exclusions", "verified_requirements"}
            assert "gold" not in kwargs["meta"]
            calls.append(payload)
        return original(messages, task, **kwargs)

    monkeypatch.setattr(router.llm, "chat_json", capture)
    monkeypatch.setattr(router.llm, "_h_knowledge_scope_review", lambda m: {"parts": scope_parts()})
    skill = rule()
    result = router.judge(skill, RAW, "tr26485")
    assert result["decision"] == "applicable" and result["decision_overridden"]
    assert result["reported_requirements"][0]["status"] == "unknown"
    assert result["requirements"][0]["status"] == "supported"
    assert result["requirements"][0]["quotes"] == ["咳呕痰水", "哕逆恶心"]
    assert result["exclusions"][0]["status"] == "unknown"
    assert result["requirements"][1:] == result["reported_requirements"][1:]
    assert result["scope_recheck"]["parts"] == scope_parts()
    assert result["active_branches"] == []  # Optional patient branches are not inferred by the scope recheck.
    assert router.judge(skill, RAW, "same-description-other-id") == result and len(calls) == 1


@pytest.mark.parametrize("mode,expected", [("missing_fact", "uncertain"), ("negative_fact", "uncertain"),
                                         ("contradiction", "not_applicable"), ("present_exclusion", "not_applicable"),
                                         ("invalid_quote", "uncertain"), ("scope_contradicted", "not_applicable")])
def test_scope_recheck_cannot_bypass_other_patient_evidence(router, monkeypatch, mode, expected):
    unknown_scope(router, monkeypatch)
    original = router.llm._h_knowledge_applicability
    skill = rule()
    if mode == "negative_fact":
        skill.details["required_conditions"] = ["已明确排除某风险"]

    def altered(meta):
        data = original(meta)
        if mode in ("missing_fact", "negative_fact"):
            data["requirements"][1].update(status="unknown", quotes=[])
        elif mode == "contradiction":
            data["requirements"][1]["status"] = "contradicted"
        elif mode == "present_exclusion":
            data["exclusions"][0].update(status="present", quotes=[RAW[:4]])
        elif mode == "invalid_quote":
            data["requirements"][0]["quotes"] = ["not-in-case"]
        else:
            data["requirements"][0].update(status="contradicted", quotes=[RAW[:4]])
        return data

    monkeypatch.setattr(router.llm, "_h_knowledge_applicability", altered)
    monkeypatch.setattr(router.llm, "_h_knowledge_scope_review", lambda m: pytest.fail("Must not bypass patient evidence"))
    result = router.judge(skill, RAW)
    assert result["decision"] == expected and result["scope_recheck"] is None


@pytest.mark.parametrize("prefix", ["仅小儿的", "已明确排除过敏风险的"])
def test_actual_population_and_negative_prerequisites_stay_unknown(router, monkeypatch, prefix):
    unknown_scope(router, monkeypatch)
    skill = rule()
    skill.details["applicability"] = prefix + "胃系病症；不用于无胃系症状或纯表证者。"
    parts = scope_parts()
    parts[0].update(text=prefix, kind="patient_condition", status="unknown", quotes=[], reason="原文未提供此必要事实。")
    monkeypatch.setattr(router.llm, "_h_knowledge_scope_review", lambda m: {"parts": parts})
    result = router.judge(skill, RAW)
    assert result["decision"] == "uncertain" and result["scope_recheck"]["ok"]


@pytest.mark.parametrize("mode,expected", [("unknown_patient", "uncertain"), ("contradicted_patient", "not_applicable"),
                                         ("present_exclusion", "not_applicable")])
def test_scope_parts_aggregate_patient_and_exclusion_evidence(router, monkeypatch, mode, expected):
    unknown_scope(router, monkeypatch)
    parts = scope_parts()
    if mode == "unknown_patient":
        parts[1].update(status="unknown", quotes=[])
    elif mode == "contradicted_patient":
        parts[1]["status"] = "contradicted"
    else:
        parts[2].update(status="present", quotes=[RAW[:4]])
    monkeypatch.setattr(router.llm, "_h_knowledge_scope_review", lambda m: {"parts": parts})
    assert router.judge(rule(), RAW)["decision"] == expected


@pytest.mark.parametrize("separator,expected", [("；", "applicable"), ("且未明确的患者条件", "uncertain")])
def test_only_punctuation_can_be_ignored_as_nonsemantic_scope_parts(router, monkeypatch, separator, expected):
    unknown_scope(router, monkeypatch)
    skill, parts = rule(), scope_parts()
    skill.details["applicability"] = "成人与小儿的胃系病症" + separator + "不用于无胃系症状或纯表证者。"
    parts[1]["text"] = "胃系病症"
    parts.insert(2, {"text": separator, "kind": "unclear", "status": "unknown", "quotes": [], "reason": "未分类片段。"})
    monkeypatch.setattr(router.llm, "_h_knowledge_scope_review", lambda m: {"parts": parts})
    assert router.judge(skill, RAW)["decision"] == expected


@pytest.mark.parametrize("bad", ["dropped_text", "reordered", "fake_quote", "unsupported_absence", "wrong_kind",
                                "missing_scope_quote", "empty", "not_a_list"])
def test_invalid_scope_recheck_never_promotes_unknown_to_supported(router, monkeypatch, bad):
    unknown_scope(router, monkeypatch)
    parts = scope_parts()
    if bad == "dropped_text":
        parts.pop(2)
    elif bad == "reordered":
        parts.reverse()
    elif bad == "fake_quote":
        parts[1]["quotes"] = ["患者年龄十五岁"]
    elif bad == "unsupported_absence":
        parts[2]["status"] = "absent"
    elif bad == "wrong_kind":
        parts[1]["kind"] = ["patient_condition"]
    elif bad == "missing_scope_quote":
        parts[0]["quotes"] = []
    else:
        parts = [] if bad == "empty" else None
    monkeypatch.setattr(router.llm, "_h_knowledge_scope_review", lambda m: {"parts": parts})
    result = router.judge(rule(), RAW)
    assert result["decision"] == "uncertain"
    assert not result["scope_recheck"]["ok"] and result["scope_recheck"]["validation_errors"]
    assert result["requirements"] == result["reported_requirements"]


@pytest.mark.parametrize("task", ["knowledge_branch_conditions", "knowledge_scope_review"])
def test_recheck_transport_failure_is_retryable_not_cached_rejection(router, monkeypatch, task):
    disputed_review(router, monkeypatch)
    unknown_scope(router, monkeypatch)
    original = router.llm.chat_json

    def failing(messages, actual_task, **kwargs):
        if actual_task == task:
            return LLMResult(ok=False, request_failed=True, error="transport failed")
        return original(messages, actual_task, **kwargs)

    monkeypatch.setattr(router.llm, "chat_json", failing)
    with pytest.raises(LLMRequestError):
        if task == "knowledge_branch_conditions":
            router.review_definition(rule())
        else:
            router.judge(rule(), RAW)
    assert not router._definitions and not router._judgments


def test_existing_blocking_issue_is_not_overridden_by_condition_recheck(router, monkeypatch):
    disputed_review(router, monkeypatch)
    original = router.llm._h_knowledge_skill_review

    def blocking(meta):
        data = original(meta)
        data["issues"] = [{"field": "text", "quote": meta["skill"]["text"], "severity": "blocking",
                           "code": "misplaced_condition", "reason": "实际必要事实被误放置。"}]
        return data

    monkeypatch.setattr(router.llm, "_h_knowledge_skill_review", blocking)
    monkeypatch.setattr(router.llm, "_h_knowledge_branch_conditions", lambda m: pytest.fail("Blocking issue remains"))
    result = router.review_definition(rule())
    assert not result["ok"] and result["blocking_issues"]
