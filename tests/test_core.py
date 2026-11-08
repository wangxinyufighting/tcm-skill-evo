"""Unit tests for the core, data-independent pieces.  Run:  python -m pytest -q"""
import os
import sys
import copy
import json
from types import SimpleNamespace

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tcmskill.config import load_config, to_cfg  # noqa: E402
from tcmskill.data import clean_desc  # noqa: E402
from tcmskill.herbs import HerbLexicon, IncompatibilityChecker, fmm  # noqa: E402
from tcmskill.metrics import at_k, paired_bootstrap, prf, soft_prf  # noqa: E402
from tcmskill.learning import CreditAssigner, ExperienceRecord, Gate  # noqa: E402
from tcmskill.skills import Skill, SkillMatcher, Trigger, merge_skills, skill_from_json  # noqa: E402

VOCAB = ["甘草", "炙甘草", "黄芪", "附子", "白附子", "麦门冬", "麦冬", "半夏", "法半夏", "海藻", "人参", "五灵脂", "茯苓", "百合", "杏", "杏仁", "桑", "桑白皮", "生地黄", "山楂", "神曲", "麦芽", "石斛", "柴胡"]
KB = os.path.join(ROOT, "kb")


@pytest.fixture(scope="module")
def lex():
    return HerbLexicon(VOCAB, os.path.join(KB, "herb_aliases.json"), merge_processed=True)


def test_fmm_longest_match():
    lexicon = {"附子": "附子", "白附子": "白附子", "杏": "杏", "杏仁": "杏仁"}
    assert fmm("白附子 杏仁", lexicon, 3) == ["白附子", "杏仁"]


def test_tokenize_fixes_substring_artifacts(lex):
    text = "白附子(炮)  黄耆三两  桑根白皮  厚朴(姜汁炒)  麦门冬"
    assert lex.tokenize(text, "ptm") == ["白附子", "附子", "桑", "麦门冬"] or "附子" in lex.tokenize(text, "ptm")
    out = lex.tokenize(text, "fmm_alias")
    assert "附子" not in out and "白附子" in out  # artifact removed
    assert "黄芪" in out  # classical name recovered
    assert "桑白皮" in out and "桑" not in out
    assert "麦冬" in out  # synonym merged


@pytest.mark.parametrize(
    "raw,expected",
    [("生黄芪15g", ["黄芪"]), ("炙甘草 6克", ["甘草"]), ("百合30g", ["百合"]), ("法半夏", ["半夏"]), ("焦三仙", ["山楂", "神曲", "麦芽"]), ("金钗石斛", ["石斛"]), ("醋柴胡", ["柴胡"]), ("不存在的药", [])],
)
def test_normalize(lex, raw, expected):
    assert lex.normalize(raw) == expected


def test_incompatibility(lex):
    inc = IncompatibilityChecker(os.path.join(KB, "incompatibility.json"), lex)
    v = inc.violations(["甘草", "海藻", "茯苓"])
    assert any(x["type"] == "十八反" for x in v)
    assert inc.violations(["人参", "五灵脂"])[0]["type"] == "十九畏"
    assert inc.violations(["附子", "半夏"])  # 药典：附子 不宜与 半夏 同用
    kept, removed = inc.filter(["甘草", "茯苓", "海藻"])
    assert kept == ["甘草", "茯苓"] and removed


def test_clean_desc():
    s = "①《温病条辨》：温病下后，大便溏甚。②《中医杂志》(1965；12：19)：伤寒肠出血。"
    out = clean_desc(s)
    assert "《" not in out and "1965" not in out and "温病下后" in out and "伤寒肠出血" in out


def test_metrics():
    assert prf(["a", "b"], ["b", "c"]) == (0.5, 0.5, 0.5)
    p, r, n = at_k(["a", "b", "c"], ["a", "c"], 2)
    assert p == 0.5 and r == 0.5 and 0 < n < 1
    same = lambda x, y: {x, y} == {"a", "z"}
    assert soft_prf(["z"], ["a"], same, 0.5)[2] == pytest.approx(0.5)
    t = paired_bootstrap([1.0] * 30, [0.0] * 30, n=200)
    assert t["diff"] == 1.0 and t["p_value"] < 0.01


class _Store:
    """Minimal stand-in for SeenStore (uniform IDF)."""

    n = 0

    def idf(self, term):
        return 1.0


def _matcher():
    return SkillMatcher(_Store(), to_cfg({"opt_weight": 0.3, "exclude_penalty": 0.7, "fire": 0.5, "mod_if_ratio": 1.0}))


def test_match_and_execute():
    m = _matcher()
    s = Skill(branch="formula", triggers=[Trigger(required=["恶寒", "无汗"], optional=["咳嗽"], exclude=["汗出"])], core=[{"name": "麻黄", "role": "君"}, {"name": "桂枝", "role": "臣"}], mods=[{"if": ["咳嗽"], "add": ["杏仁"], "remove": []}])
    assert m.match(s, "恶寒无汗，咳嗽", set()) == pytest.approx(1.0)
    assert m.match(s, "恶寒无汗，汗出", set()) < 0.5  # exclude penalty
    assert m.execute(s, "恶寒无汗，咳嗽", set()) == ["麻黄", "桂枝", "杏仁"]
    assert m.execute(s, "头痛", set()) == []


def test_skill_from_json_grounding(lex):
    obj = {"branch": "formula", "name": "测试", "trigger": {"required": ["咳嗽", "自造的词"], "optional": []}, "core_herbs": [{"name": "生黄芪", "role": "君"}, {"name": "茯苓", "role": "臣"}], "modifications": [{"if": ["咳嗽"], "add": ["杏仁"]}]}
    s, why = skill_from_json(obj, lex, ["久咳嗽，气短"])
    assert why == "ok" and s.triggers[0].required == ["咳嗽"]  # ungrounded term dropped
    assert s.core_herbs() == ["黄芪", "茯苓"]
    bad, why = skill_from_json({"branch": "formula", "trigger": {"required": ["不存在"]}, "core_herbs": ["甘草"]}, lex, ["咳嗽"])
    assert bad is None and why == "ungrounded_trigger"


def test_merge_multi_trigger():
    a = Skill(id="F1", branch="formula", triggers=[Trigger(required=["咳嗽"])], core=[{"name": "杏仁", "role": ""}])
    b = Skill(id="F2", branch="formula", triggers=[Trigger(required=["腹痛"])], core=[{"name": "杏仁", "role": ""}])
    m = merge_skills(a, b)
    assert m.id == "F1" and len(m.triggers) == 2  # 异病同治: same formula, two trigger patterns


@pytest.mark.parametrize("mode", ["herb", "episode"])
@pytest.mark.parametrize("used", [[], ["M1"]])
def test_failed_predictions_do_not_update_skill_credit(mode, used):
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"), overrides=[f"skills.credit={mode}"])
    credit = CreditAssigner(cfg.skills, _matcher())
    credit.base_f1 = {"0": 0.4}
    credit.base_prec = {"0|modification": 0.3}
    skill = Skill(id="M1", branch="modification", triggers=[Trigger(required=["cough"])], add=["herb-a"])
    before = copy.deepcopy(skill)
    record = ExperienceRecord(cid="c1", cat=0, text="cough", symptoms=["cough"], gt=["herb-a"],
                              used=used, f1=0.0, ok=False)
    credit.observe(SimpleNamespace(cid="c1", cat=0, symptoms=["cough"]), "cough", record, [skill], 1)
    assert skill == before
    assert credit.pending == {}
    assert credit.base_f1 == {"0": 0.4}
    assert credit.base_prec == {"0|modification": 0.3}


def test_valid_zero_score_prediction_still_gets_credit_updates():
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    credit = CreditAssigner(cfg.skills, _matcher())
    credit.base_f1 = {"0": 0.4}
    skill = Skill(id="M1", branch="modification", triggers=[Trigger(required=["cough"])], add=["herb-a"])
    record = ExperienceRecord(cid="c1", cat=0, text="cough", symptoms=["cough"], gt=["herb-a"],
                              pred=["herb-b"], f1=0.0, ok=True)
    credit.observe(SimpleNamespace(cid="c1", cat=0, symptoms=["cough"]), "cough", record, [skill], 1)
    assert skill.n_ret == 1 and credit.pending["M1"]
    assert credit.base_f1["0"] == pytest.approx(0.38)


def test_gate_rng_roundtrips_through_json():
    cfg = load_config(os.path.join(ROOT, "configs/default.yaml"))
    gate = Gate(cfg.skills.gate, seed=42)
    records = [ExperienceRecord(cid=f"c{i}", cat=0, text="", symptoms=[], gt=["herb-a"], f1=1.0)
               for i in range(20)]
    gate.select(records)
    state = json.loads(json.dumps(gate.state_dict()))
    restored = Gate(cfg.skills.gate, seed=42)
    restored.load_state_dict(state)
    assert [r.cid for r in restored.select(records)] == [r.cid for r in gate.select(records)]
