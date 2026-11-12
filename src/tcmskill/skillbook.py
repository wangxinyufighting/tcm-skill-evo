"""Skill book: an evolving, LLM-written prescribing skill document.

One practice step = one minibatch of training cases (retrieval always excludes the case itself):

  1. group rollouts     a  agent (T=0) with the current skill book      b  agent re-sampled (T>0)
                        c  copy the most similar case C1                 d  kNN vote of the neighbours
                        every rollout is scored against the original formula (F1, P, R, length)
  2. semantic advantage an optimizer LLM compares high- and low-scoring rollouts of the same case
                        (Training-Free GRPO; ExpeL success/failure contrast) and proposes at most
                        k bounded edits (add / replace / delete) to the skill book (SkillOpt; ACE)
  3. evidence filter    an edit must cite >= min_support practice cases; one- or two-case "rules" are dropped
  4. validation gate    every edit is tested on its own against the current book (per_edit; SkillOpt) on
                        held-out training cases whose prompt it changes: a screening stage on the first
                        n_screen of them (GEPA-style minibatch), then all of them (paired mean ΔF1 >= min_delta
                        and t >= t_min); rejected edits and their ΔF1 are shown to the optimizer afterwards.
                        gate_mode=joint (all edits of a step tested as one bundle) is the first version, kept
                        for ablations
  5. confirmation       an edit that passed the gate is re-tested on a second, disjoint held-out set
                        (n_confirm) and kept only if it does not hurt there either (counters the winner's curse
                        of trying many edits on the same gate cases)

Rules are general (always shown, system prompt) or scoped (shown when one of their keywords occurs
in the case description). The policy prompt itself is neutral: every strategy the agent follows -
when to copy a precedent, how many herbs, how to use the neighbours, disease-specific habits - has
to be learned into the skill book.

A request that still fails after the client's own retries is never scored as an empty prescription
(F1 = 0 would bias the rollouts and the gate): failed cases are re-run, and if they keep failing the
step is not committed and the run stops; rerunning the same command resumes from that step.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass, field
from functools import partial

import numpy as np

from . import prompts as P
from .decision import KNNVoter, fuse
from .llm import FatalLLMError, LLMRequestError
from .metrics import prf
from .utils import dedup_keep_order, truncate

LOG = logging.getLogger(__name__)
_RID = re.compile(r"R\d+")


def _bigrams(s: str) -> set:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _similar(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / max(1, len(A | B))


@dataclass
class Rule:
    id: str
    text: str
    scope: list = field(default_factory=list)
    added: int = 0
    uses: int = 0
    adv_sum: float = 0.0
    gain: float | None = None  # held-out ΔF1 measured by the gate when the rule was accepted
    details: dict = field(default_factory=dict)

    def render(self) -> str:
        text = f"{self.id}（适用：{'、'.join(self.scope)}）{self.text}" if self.scope else f"{self.id}. {self.text}"
        return text + ("\n" + json.dumps(self.details, ensure_ascii=False) if self.details else "")

    def render_stats(self) -> str:
        where = "、".join(self.scope) if self.scope else "通用"
        st = f"被引用 {self.uses} 次，平均优势 {self.adv_sum / self.uses:+.3f}" if self.uses else "尚未被引用"
        if self.gain is not None:
            st += f"，加入时留出病例 ΔF1 {self.gain:+.3f}"
        return f"{self.id}（{where}｜{st}）{self.text}"


class SkillBook:
    def __init__(self, max_general: int = 20, max_scoped: int = 80, max_chars: int = 100):
        self.rules: list[Rule] = []
        self.next_id = 1
        self.max_general, self.max_scoped, self.max_chars = max_general, max_scoped, max_chars

    # ------------------------------------------------------------------ views
    def general(self) -> list[Rule]:
        return [r for r in self.rules if not r.scope]

    def get(self, rid: str) -> Rule | None:
        return next((r for r in self.rules if r.id == rid), None)

    def scoped_for(self, text: str, symptoms=(), k: int = 6) -> list[Rule]:
        sym = set(symptoms or [])
        hit = [r for r in self.rules if r.scope and any(w in text or w in sym for w in r.scope)]
        hit.sort(key=lambda r: (-(r.adv_sum / r.uses if r.uses else 0.0), int(r.id[1:])))
        return hit[:k]

    def render_general(self) -> str:
        return "\n".join(r.render() for r in self.general())

    def render_for_optimizer(self) -> str:
        return "\n".join(r.render_stats() for r in self.rules)

    def signature(self) -> str:
        rows = [(r.id, r.text, r.scope, r.details) if r.details else (r.id, r.text, r.scope) for r in self.rules]
        blob = json.dumps(rows, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(blob.encode("utf-8")).hexdigest()[:10]

    def to_markdown(self) -> str:
        g = self.general()
        s = [r for r in self.rules if r.scope]
        lines = ["## 通用技能", ""] + [f"- **{r.id}** {r.text}  `引用 {r.uses}｜平均优势 {(r.adv_sum / r.uses) if r.uses else 0:+.3f}｜第 {r.added} 步加入`" for r in g]
        lines += ["", "## 病证专项技能", ""] + [f"- **{r.id}**（{'、'.join(r.scope)}）{r.text}  `引用 {r.uses}｜平均优势 {(r.adv_sum / r.uses) if r.uses else 0:+.3f}｜第 {r.added} 步加入`" for r in s]
        for r in self.rules:
            if r.details:
                lines += ["", f"### {r.id} 证据与适用边界", "", "```json", json.dumps(r.details, ensure_ascii=False, indent=2), "```"]
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------ edits
    def _clean_text(self, t) -> str:
        t = re.sub(r"^\s*R\d+[.．、:：]?\s*", "", str(t or "")).strip()
        return truncate(t, self.max_chars)

    @staticmethod
    def _clean_scope(sc) -> list:
        if isinstance(sc, str):
            sc = re.split(r"[、，,；;\s]+", sc)
        out = []
        for w in sc or []:
            w = str(w).strip()
            if 1 <= len(w) <= 8 and w not in ("通用", "全部", "所有") and w not in out:
                out.append(w)
        return out[:4]

    def apply(self, edits: list[dict], step: int) -> tuple["SkillBook", list[dict], list[tuple[dict, str]]]:
        new = copy.deepcopy(self)
        applied, skipped = [], []
        for e in edits:
            op = str(e.get("op", "")).lower()
            if op == "delete":
                r = new.get(str(e.get("id", "")))
                if r is None:
                    skipped.append((e, "unknown_id"))
                    continue
                new.rules.remove(r)
                applied.append({"op": "delete", "id": r.id, "text": r.text, "scope": r.scope})
                continue
            text = new._clean_text(e.get("text"))
            if len(text) < 6:
                skipped.append((e, "empty_text"))
                continue
            scope = new._clean_scope(e.get("scope"))
            if op == "replace":
                r = new.get(str(e.get("id", "")))
                if r is None:
                    skipped.append((e, "unknown_id"))
                    continue
                if any(o is not r and _similar(o.text, text) >= 0.7 for o in new.rules):
                    skipped.append((e, "duplicate"))
                    continue
                applied.append({"op": "replace", "id": r.id, "old": r.text, "text": text, "scope": scope})
                r.text, r.scope, r.added, r.uses, r.adv_sum = text, scope, step, 0, 0.0
                r.details = copy.deepcopy(e.get("details") or {})
                if r.details:
                    applied[-1]["details"] = r.details
                continue
            if op != "add":
                skipped.append((e, "unknown_op"))
                continue
            if any(_similar(o.text, text) >= 0.7 for o in new.rules):
                skipped.append((e, "duplicate"))
                continue
            if (not scope and len(new.general()) >= new.max_general) or (scope and len(new.rules) - len(new.general()) >= new.max_scoped):
                skipped.append((e, "book_full"))
                continue
            r = Rule(id=f"R{new.next_id}", text=text, scope=scope, added=step,
                     details=copy.deepcopy(e.get("details") or {}))
            new.next_id += 1
            new.rules.append(r)
            applied.append({"op": "add", "id": r.id, "text": text, "scope": scope})
            if r.details:
                applied[-1]["details"] = r.details
        return new, applied, skipped

    def state_dict(self) -> dict:
        return {"rules": [asdict(r) for r in self.rules], "next_id": self.next_id,
                "limits": [self.max_general, self.max_scoped, self.max_chars]}

    @classmethod
    def from_state(cls, d: dict) -> "SkillBook":
        lim = d.get("limits", [20, 80, 100])
        b = cls(*lim)
        b.rules = [Rule(**r) for r in d.get("rules", [])]
        b.next_id = int(d.get("next_id", len(b.rules) + 1))
        return b


# ============================================================================ policy
class SkillPolicy:
    """The agent: neutral prompt + skill book + k similar cases (with similarity)."""

    def __init__(self, res, voter: KNNVoter, k_cases: int = 5, scoped_per_case: int = 6, max_herbs: int = 30, vocab_min_freq: int = 1,
                 retry_passes: int = 2, retry_wait: float = 30.0):
        self.res, self.voter, self.k, self.k_scoped, self.max_herbs = res, voter, k_cases, scoped_per_case, max_herbs
        labels = [h for h in res.labels if res.herb_freq.get(h, 0) >= vocab_min_freq]
        self.vocab = P.vocab_block(labels)
        self.retry_passes, self.retry_wait = max(0, int(retry_passes)), max(0.0, float(retry_wait))
        self.n_retried = 0  # requests re-run after the client's own retries were exhausted

    def messages(self, book: SkillBook, case, kv: dict) -> tuple[list[dict], list[Rule], list[Rule]]:
        res = self.res
        general = book.general()
        system = P.SKILL_RX_SYSTEM + self.vocab + ("\n\n【开方技能（通用）】\n" + book.render_general() if general else "")
        nb = kv.get("nb", [])[: self.k]
        cands = "\n".join(P.render_candidate(i + 1, res.text(res.cases[cid]), res.cases[cid].herbs, sim) for i, (cid, sim) in enumerate(nb))
        scoped = book.scoped_for(res.text(case), case.symptoms, self.k_scoped)
        user = P.skill_rx_user(res.text(case), cands, "\n".join(r.render() for r in scoped))
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], general, scoped

    def prompt_hash(self, book: SkillBook, case, kv: dict) -> str:
        msgs, _, _ = self.messages(book, case, kv)
        return hashlib.md5(json.dumps(msgs, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _answer(self, book: SkillBook, llm, item) -> dict:
        res = self.res
        case, kv = item
        msgs, general, scoped = self.messages(book, case, kv)
        shown = general + scoped
        nb = kv.get("nb", [])[: self.k]
        meta = {"cid": case.cid, "rules": [r.text for r in shown], "rule_ids": [r.id for r in shown],
                "cases": [{"herbs": res.cases[cid].herbs, "sim": sim} for cid, sim in nb],
                "fallback": res.cases[nb[0][0]].herbs if nb else ["甘草"]}
        r = llm.chat_json(msgs, task="skill_rx", meta=meta)
        out = {"cid": case.cid, "herbs": [], "used": [], "ok": False, "error": r.error, "analysis": "",
               "json_status": getattr(r, "json_status", "unknown"), "finish_reason": getattr(r, "finish_reason", None),
               "request_failed": bool(getattr(r, "request_failed", False))}
        d = r.data if r.ok and isinstance(r.data, dict) else None
        if d is None:
            return out
        names = d.get("herbs") or []
        if isinstance(names, str):
            names = re.split(r"[、，,；;\s]+", names)
        names = [(n.get("name") or n.get("药名") or "") if isinstance(n, dict) else str(n) for n in names if n]
        herbs, oov = res.lexicon.normalize_list(names)
        herbs = dedup_keep_order(herbs)[: self.max_herbs]
        valid = {x.id for x in shown}
        used = dedup_keep_order([u for u in _RID.findall(json.dumps(d.get("used_rules") or [], ensure_ascii=False)) if u in valid])
        out.update(herbs=herbs, used=used, ok=bool(herbs), oov=oov, analysis=truncate(str(d.get("analysis") or ""), 80),
                   error=None if herbs else "empty_prescription")
        return out

    def run(self, book: SkillBook, cases: list, kvs: list, llm, desc: str = "skill_rx") -> list[dict]:
        """One answer per case. Invalid JSON / empty answers are the policy's own failures and score 0; a request
        that failed at the transport level (after the client's retries) is re-run `retry_passes` times,
        `retry_wait` s apart, and raises LLMRequestError if it never succeeds - it is never scored."""
        items = list(zip(cases, kvs))
        one = partial(self._answer, book, llm)
        outs = llm.map(one, items, desc=desc)
        for p in range(self.retry_passes):
            bad = [i for i, o in enumerate(outs) if o["request_failed"]]
            if not bad:
                break
            LOG.warning("%s：%d 个请求重试耗尽，%.0fs 后重跑这些病例（第 %d/%d 轮）", desc, len(bad), self.retry_wait, p + 1, self.retry_passes)
            time.sleep(self.retry_wait)
            self.n_retried += len(bad)
            for i, o in zip(bad, llm.map(one, [items[i] for i in bad], desc=f"{desc}_retry{p + 1}")):
                outs[i] = o
        bad = [o["cid"] for o in outs if o["request_failed"]]
        if bad:
            raise LLMRequestError(f"{desc}：{len(bad)}/{len(items)} 个请求多次重试后仍失败（如 {', '.join(map(str, bad[:3]))}）")
        return outs


# ============================================================================ optimizer
def _score(herbs, gt) -> dict:
    p, r, f = prf(herbs, gt)
    return {"f1": round(f, 3), "p": round(p, 3), "r": round(r, 3), "n": len(herbs)}


def build_groups(res, cases, kvs, outs_a, outs_b, references: bool = True) -> list[dict]:
    groups = []
    for i, (c, kv) in enumerate(zip(cases, kvs)):
        nb = kv.get("nb", [])
        roll = {"a": outs_a[i]["herbs"]}
        if references:  # non-LLM alternatives the agent could have taken: copy C1 / vote of the neighbours
            roll["c"] = list(res.cases[nb[0][0]].herbs) if nb else []
            roll["d"] = fuse([], kv, a=0.0, length="knn", add_thr=0.0)
        used = {"a": outs_a[i]["used"]}
        if outs_b is not None:
            roll["b"] = outs_b[i]["herbs"]
            used["b"] = outs_b[i]["used"]
        sc = {k: _score(v, c.herbs) for k, v in roll.items()}
        f = [s["f1"] for s in sc.values()]
        groups.append({"case": c, "kv": kv, "roll": roll, "score": sc, "used": used, "gap": max(f) - min(f),
                       "regret": max(f) - sc["a"]["f1"], "mean": float(np.mean(f))})
    return groups


def update_rule_stats(book: SkillBook, groups: list[dict]) -> None:
    """SkeMex-style utility: when a rule is cited, credit it with the rollout's advantage over the group mean."""
    for g in groups:
        for k, ids in g["used"].items():
            adv = g["score"][k]["f1"] - g["mean"]
            for rid in ids:
                r = book.get(rid)
                if r is not None:
                    r.uses += 1
                    r.adv_sum += adv


def render_groups(res, groups: list[dict], k_show: int = 3) -> str:
    names = {"a": "a 智能体", "b": "b 重新采样", "c": "c 照抄 C1", "d": "d 多案投票"}
    blocks = []
    for j, g in enumerate(groups, 1):
        c = g["case"]
        nb = g["kv"].get("nb", [])[:k_show]
        neigh = "；".join(f"C{i + 1} sim{s:.2f}「{truncate(res.text(res.cases[cid]), 28)}」" for i, (cid, s) in enumerate(nb))
        lines = [f"#{j} 主治：{truncate(res.text(c), 90)}", f"   相似医案：{neigh or '无'}", f"   原方（{len(c.herbs)} 味）：{'、'.join(c.herbs)}"]
        for k in ("a", "b", "c", "d"):
            if k not in g["roll"]:
                continue
            s = g["score"][k]
            used = f"（用了 {'、'.join(g['used'][k])}）" if g["used"].get(k) else ""
            lines.append(f"   {names[k]}{used}：{'、'.join(g['roll'][k]) or '（空）'}｜F1 {s['f1']:.2f} P {s['p']:.2f} R {s['r']:.2f}｜{s['n']} 味")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def pick_informative(groups: list[dict], n: int, min_gap: float, balanced: bool = True) -> list[dict]:
    """Groups with a clear winner and loser (std(r) > 0 in Training-Free GRPO).
    balanced (default): alternate cases where the agent could have done better (high regret) with cases where
    the agent itself was best and an alternative would have hurt - otherwise the optimizer only sees cases in
    which copying / voting won and over-generalises them (selection bias seen in sk_v2).
    balanced=False: mostly regret cases plus n//4 positives (first version).
    Without any alternative rollout (ablation "no contrast") fall back to the agent's worst cases."""
    if all(len(g["roll"]) == 1 for g in groups):
        return sorted(groups, key=lambda g: g["score"]["a"]["f1"])[:n]
    pool = [g for g in groups if g["gap"] >= min_gap]
    worse = sorted([g for g in pool if g["regret"] > 0], key=lambda g: -g["regret"])
    better = sorted([g for g in pool if g["regret"] <= 0], key=lambda g: -g["gap"])
    if not balanced:
        n_pos = min(len(better), max(1, n // 4))
        return (worse[: n - n_pos] + better[:n_pos])[:n]
    out = []
    while len(out) < n and (worse or better):
        for lst in (worse, better):
            if lst and len(out) < n:
                out.append(lst.pop(0))
    return out


def case_record(g: dict) -> dict:
    """Per-case outcome of one practice step, kept for the optimizer's cumulative statistics."""
    nb = g["kv"].get("nb", [])
    s1 = float(nb[0][1]) if nb else 0.0
    rec = {"cid": g["case"].cid, "sim1": round(s1, 3), "tie": bool(nb) and sum(1 for _, x in nb[:5] if abs(x - s1) < 1e-9) > 1,
           "n_gt": len(g["case"].herbs)}
    for k, sc in g["score"].items():
        rec[k], rec["n_" + k] = round(float(sc["f1"]), 4), int(sc["n"])
    return rec


_BANDS = [("≥0.9 且无并列", lambda r: r["sim1"] >= 0.9 and not r["tie"]),
          ("≥0.9 且有并列（几个医案同样相似）", lambda r: r["sim1"] >= 0.9 and r["tie"]),
          ("0.6–0.9", lambda r: 0.6 <= r["sim1"] < 0.9), ("0.45–0.6", lambda r: 0.45 <= r["sim1"] < 0.6),
          ("<0.45", lambda r: r["sim1"] < 0.45)]


def _stats_line(rows: list[dict]) -> str:
    mean = lambda k: float(np.mean([r[k] for r in rows]))
    parts = [f"智能体 a {mean('a'):.2f}"]
    for k, name in (("c", "照抄 C1"), ("d", "多案投票")):
        if all(k in r for r in rows):
            better = sum(r[k] > r["a"] + 1e-9 for r in rows)
            worse = sum(r[k] < r["a"] - 1e-9 for r in rows)
            parts.append(f"{name} {mean(k):.2f}（比 a 好 {better} 例、差 {worse} 例）")
    parts.append(f"a 平均 {mean('n_a'):.1f} 味 / 原方 {mean('n_gt'):.1f} 味")
    return f"{len(rows)} 例：" + "；".join(parts)


def render_stats(batch: list[dict], hist: list[dict], bands: bool = True) -> str:
    """Base rates for the optimizer (mean F1 per strategy, how often an alternative beats the agent), for this
    batch and, by similarity of the nearest precedent, for every practice case seen so far."""
    lines = ["本批全部 " + _stats_line(batch)] if batch else []
    if bands and hist:
        lines.append(f"累计练习（共 {len(hist)} 例，a 为当时技能下的作答），按最相似医案 C1 的相似度分段：")
        for name, f in _BANDS:
            rows = [r for r in hist if f(r)]
            if rows:
                lines.append(f"- C1 相似度 {name}：" + _stats_line(rows))
    return "\n".join(lines)


_CITE = re.compile(r"(?:#|＃|病例)\s*(\d+(?:\s*[、,，和及与/]\s*(?:#|＃)?\s*\d+)*)")


_COUNTER = re.compile(r"反例|反证|例外")


def edit_evidence(edit: dict, n_cases: int) -> tuple[int, int]:
    """(supporting, counter-example) practice cases an edit cites in its `why`: "支持 #2 #5 #9；反例 #4：...".
    Counter-examples are the case numbers listed right after 反例 / 反证 / 例外; every other #k is support.
    Cases are #1..n_cases."""
    why = str(edit.get("why") or "")
    nums = lambda m: {int(x) for x in re.findall(r"\d+", m.group(1)) if 1 <= int(x) <= n_cases}
    cnt, spans = set(), []
    for k in _COUNTER.finditer(why):
        m = _CITE.search(why, k.end())
        if m is not None and re.fullmatch(r"[\s:：为是有如]*", why[k.end(): m.start()]):
            cnt |= nums(m)
            spans.append((m.start(), m.end()))
    sup = set()
    for m in _CITE.finditer(why):
        if not any(a <= m.start() and m.end() <= b for a, b in spans):
            sup |= nums(m)
    return len(sup - cnt), len(cnt)


_CLAIM = re.compile(r"好\s*(\d+)\s*例[^\d好差]{0,6}差\s*(\d+)\s*例")


def history_evidence(edit: dict, stats_text: str) -> tuple[int, int]:
    """Better / worse counts the optimizer quotes from the 【统计概况】 table ("好 5 例、差 0 例") as evidence for a rule.
    Only pairs that really appear in the table it was shown are counted."""
    s = c = 0
    seen = set()
    for x, y in _CLAIM.findall(str(edit.get("why") or "")):
        if (x, y) not in seen and f"好 {x} 例、差 {y} 例" in (stats_text or ""):
            seen.add((x, y))
            s, c = s + int(x), c + int(y)
    return s, c


def edit_support(edit: dict, n_cases: int) -> int:
    """Number of distinct practice cases (#k, 1..n_cases) an edit cites as supporting evidence."""
    return edit_evidence(edit, n_cases)[0]


def parse_edits(data, k: int) -> tuple[list[dict], str]:
    if not isinstance(data, dict):
        return [], ""
    edits = data.get("edits") or []
    edits = [e for e in edits if isinstance(e, dict) and str(e.get("op", "")).lower() in ("add", "replace", "delete")]
    return edits[:k], truncate(str(data.get("analysis") or ""), 200)


# ============================================================================ gate
def paired_stats(new: list[float], old: list[float]) -> tuple[float, float]:
    d = np.asarray(new, float) - np.asarray(old, float)
    if len(d) == 0:
        return 0.0, 0.0
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    t = m / (sd / math.sqrt(len(d))) if sd > 0 else (math.inf if m > 0 else 0.0)
    return m, t


# ============================================================================ practice loop
class Evolver:
    def __init__(self, res, cfg, run_dir: str, llm0, llm_explore, llm_opt):
        import os

        sc = cfg.skillbook
        self.res, self.cfg, self.sc, self.run_dir = res, cfg, sc, run_dir
        self.llm0, self.llm_explore, self.llm_opt = llm0, llm_explore, llm_opt
        train = res.select("train")
        n_p, n_s, n_c = int(sc.n_practice), int(sc.n_sel), int(sc.get("n_confirm", 0))
        if n_p + n_s + n_c > len(train):
            raise ValueError("n_practice + n_sel + n_confirm exceeds the training split")
        self.practice, self.sel = train[:n_p], train[len(train) - n_s:]
        self.confirm = train[len(train) - n_s - n_c: len(train) - n_s]   # second held-out set (confirmation)
        self.pool = train[n_p: len(train) - n_s - n_c]
        self.voter = KNNVoter(res, train, k=20)
        self.policy = SkillPolicy(res, self.voter, int(sc.k_cases), int(sc.scoped_per_case), int(cfg.prescribe.max_herbs), int(cfg.prescribe.vocab_min_freq),
                                  int(sc.get("retry_passes", 2)), float(sc.get("retry_wait", 30)))
        self.book = SkillBook(int(sc.max_general), int(sc.max_scoped), int(sc.max_rule_chars))
        self.rejected: list[dict] = []
        self.versions: list[dict] = []
        self.history: list[dict] = []  # per-case outcomes of every practice step (cumulative statistics for the optimizer)
        self.step = 0
        self.sel_cases = res.get(self.sel)
        self.sel_kvs = self.voter.votes(self.sel_cases)
        self.confirm_cases = res.get(self.confirm)
        self.confirm_kvs = self.voter.votes(self.confirm_cases) if self.confirm_cases else []
        self.state_path = os.path.join(run_dir, "state.json")
        self.log_path = os.path.join(run_dir, "evolve_log.jsonl")
        os.makedirs(os.path.join(run_dir, "versions"), exist_ok=True)

    # ------------------------------------------------------------------ persistence
    def state_dict(self) -> dict:
        return {"step": self.step, "book": self.book.state_dict(), "rejected": self.rejected, "versions": self.versions,
                "history": self.history}

    def load(self) -> bool:
        import os

        if not os.path.exists(self.state_path):
            return False
        with open(self.state_path, encoding="utf-8") as f:
            d = json.load(f)
        self.step, self.rejected, self.versions = int(d["step"]), d.get("rejected", []), d.get("versions", [])
        self.history = d.get("history", [])
        self.book = SkillBook.from_state(d["book"])
        return True

    def save(self) -> None:
        from .utils import save_json

        save_json(self.state_path, self.state_dict())

    def _restore(self, snap: dict) -> None:
        self.step, self.rejected, self.versions = int(snap["step"]), snap["rejected"], snap["versions"]
        self.history = snap.get("history", [])
        self.book = SkillBook.from_state(snap["book"])

    def cost(self) -> float:
        tot = 0.0
        for llm in {id(x): x for x in (self.llm0, self.llm_explore, self.llm_opt) if x is not None}.values():
            tot += float(llm.usage.summary().get("total", {}).get("cost_usd", 0.0))
        return round(tot, 4)

    # ------------------------------------------------------------------ gate
    def _gate_set(self, base: SkillBook, new_book: SkillBook, applied: list[dict]) -> tuple[list, list]:
        cases, kvs = list(self.sel_cases), list(self.sel_kvs)
        if new_book.render_general() != base.render_general() or int(self.sc.scoped_pool) <= 0:
            return cases, kvs
        words = set()
        for e in applied:  # scoped-only edit: add matching cases from the validation pool
            words |= set(e.get("scope") or [])
            old = base.get(e.get("id", ""))
            if old is not None:
                words |= set(old.scope)
        if not words:
            return cases, kvs
        have = set(self.sel)
        extra = []
        for cid in self.pool:
            if cid in have:
                continue
            c = self.res.cases[cid]
            t = self.res.text(c)
            if any(w in t or w in (c.symptoms or []) for w in words):
                extra.append(c)
                if len(extra) >= int(self.sc.scoped_pool):
                    break
        return cases + extra, kvs + (self.voter.votes(extra) if extra else [])

    def gate(self, base: SkillBook, new_book: SkillBook, applied: list[dict]) -> dict:
        """Paired comparison base -> new_book on the held-out cases whose prompt the edit changes.
        Two stages (GEPA-style minibatch screening): the first `n_screen` affected cases first; an edit whose
        mean ΔF1 there is <= `screen_delta` is rejected early, the others are judged on all affected cases
        (mean ΔF1 >= min_delta and paired t >= t_min). Early stopping only ever rejects, so it never adds
        false accepts; it just saves the calls that clearly useless edits would cost."""
        sc = self.sc
        cases, kvs = self._gate_set(base, new_book, applied)
        changed = [i for i, (c, kv) in enumerate(zip(cases, kvs)) if self.policy.prompt_hash(base, c, kv) != self.policy.prompt_hash(new_book, c, kv)]
        out = {"n_gate": len(cases), "n_affected": len(changed)}
        if len(changed) < int(sc.min_affected):
            return {**out, "accept": False, "reason": "too_few_affected", "delta": 0.0, "t": 0.0, "n_tested": 0}
        n1 = int(sc.get("n_screen", 0))
        stages = [changed[:n1], changed[n1:]] if 0 < n1 < len(changed) else [changed]
        f_old, f_new = [], []
        for j, idx in enumerate(stages):
            sub, skv = [cases[i] for i in idx], [kvs[i] for i in idx]
            old = self.policy.run(base, sub, skv, self.llm0, desc="gate_old")
            new = self.policy.run(new_book, sub, skv, self.llm0, desc="gate_new")
            f_old += [prf(o["herbs"], c.herbs)[2] for o, c in zip(old, sub)]
            f_new += [prf(o["herbs"], c.herbs)[2] for o, c in zip(new, sub)]
            m, t = paired_stats(f_new, f_old)
            if j == 0 and len(stages) > 1 and m <= float(sc.get("screen_delta", 0.0)):
                break
        stats = {"delta": round(m, 4), "t": round(t, 2) if math.isfinite(t) else 99.0, "n_tested": len(f_new),
                 "f_old": round(float(np.mean(f_old)), 4), "f_new": round(float(np.mean(f_new)), 4)}
        if len(f_new) < len(changed):
            return {**out, **stats, "accept": False, "reason": "screened_out"}
        acc = m >= float(sc.min_delta) and t >= float(sc.t_min)
        if not acc:
            return {**out, **stats, "accept": False, "reason": "rejected"}
        cf = self._confirm(base, new_book)
        if cf.get("tested") and cf["delta"] < float(sc.get("confirm_delta", 0.0)):
            return {**out, **stats, "confirm": cf, "accept": False, "reason": "not_confirmed"}
        return {**out, **stats, **({"confirm": cf} if cf else {}), "accept": True}

    def _confirm(self, base: SkillBook, new_book: SkillBook) -> dict:
        """Re-test an edit that passed the gate on a second, disjoint held-out set. Many edits are tried on the same
        gate cases over a run, so some pass by luck (winner's curse); a lucky edit has no reason to also help here.
        Edits that change too few confirmation prompts (narrow scoped rules) are not re-tested."""
        if not self.confirm_cases:
            return {}
        idx = [i for i, (c, kv) in enumerate(zip(self.confirm_cases, self.confirm_kvs))
               if self.policy.prompt_hash(base, c, kv) != self.policy.prompt_hash(new_book, c, kv)]
        if len(idx) < int(self.sc.min_affected):
            return {"tested": False, "n_affected": len(idx)}
        sub, skv = [self.confirm_cases[i] for i in idx], [self.confirm_kvs[i] for i in idx]
        old = self.policy.run(base, sub, skv, self.llm0, desc="confirm_old")
        new = self.policy.run(new_book, sub, skv, self.llm0, desc="confirm_new")
        f_old = [prf(o["herbs"], c.herbs)[2] for o, c in zip(old, sub)]
        f_new = [prf(o["herbs"], c.herbs)[2] for o, c in zip(new, sub)]
        m, t = paired_stats(f_new, f_old)
        return {"tested": True, "n_affected": len(idx), "delta": round(m, 4), "t": round(t, 2) if math.isfinite(t) else 99.0}

    # ------------------------------------------------------------------ one step
    def _rejected_text(self) -> str:
        rows = []
        for rej in self.rejected[-int(self.sc.max_rejected_shown):]:
            for e in rej["edits"]:
                what = e.get("text") or e.get("old") or ""
                sc = f"（适用：{'、'.join(e['scope'])}）" if e.get("scope") else ""
                joint = "（与同步其他编辑一起检验）" if len(rej["edits"]) > 1 else ""
                rows.append(f"- {e['op']} {e.get('id', '')}「{truncate(what, 60)}」{sc}：留出病例 ΔF1 {rej.get('delta', 0):+.3f}{joint}")
        return "\n".join(rows)

    def _optimize(self, msgs: list[dict], meta: dict):
        budget = int(self.sc.get("optimizer_max_tokens", 0)) or None
        r = self._opt_call(msgs, meta, budget)
        if r.ok or not getattr(r, "truncated", False):
            return r
        bigger = int(1.5 * (budget or self.llm_opt.output_budget()))
        LOG.warning("第 %d 步优化器输出被截断（思考内容也占输出额度），改用 max_tokens=%d 再试一次", self.step, bigger)
        try:
            r2 = self._opt_call(msgs, meta, bigger)
        except LLMRequestError:
            return r
        return r2 if r2.ok or not r2.request_failed else r

    def _opt_call(self, msgs: list[dict], meta: dict, budget: int | None):
        r = self.llm_opt.chat_json(msgs, task="skill_opt", meta=meta, max_tokens=budget)
        if r.request_failed and budget is not None and "max_tokens" in str(r.error):
            LOG.warning("端点不接受 max_tokens=%d，改用默认输出上限：%s", budget, str(r.error)[:160])
            return self._opt_call(msgs, meta, None)
        for p in range(self.policy.retry_passes):
            if not r.request_failed:
                break
            LOG.warning("第 %d 步优化器请求重试耗尽，%.0fs 后重试（第 %d/%d 轮）", self.step, self.policy.retry_wait, p + 1, self.policy.retry_passes)
            time.sleep(self.policy.retry_wait)
            self.policy.n_retried += 1
            r = self.llm_opt.chat_json(msgs, task="skill_opt", meta=meta, max_tokens=budget)
        if r.request_failed:
            raise LLMRequestError(f"第 {self.step} 步优化器请求多次重试后仍失败：{r.error}")
        return r

    def run_step(self, batch_ids: list, epoch: int) -> dict:
        r0 = self.policy.n_retried
        row = self._run_step(batch_ids, epoch)
        row["n_rules_after"] = len(self.book.rules)
        if self.policy.n_retried > r0:
            row["retried"] = self.policy.n_retried - r0
        return row

    def _run_step(self, batch_ids: list, epoch: int) -> dict:
        res, sc = self.res, self.sc
        self.step += 1
        cases = res.get(batch_ids)
        kvs = self.voter.votes(cases)
        a = self.policy.run(self.book, cases, kvs, self.llm0, desc=f"step{self.step}_a")
        b = self.policy.run(self.book, cases, kvs, self.llm_explore, desc=f"step{self.step}_b") if self.llm_explore is not None else None
        groups = build_groups(res, cases, kvs, a, b, bool(sc.get("references", True)))
        update_rule_stats(self.book, groups)
        mean = lambda k: round(float(np.mean([g["score"][k]["f1"] for g in groups if k in g["score"]] or [float("nan")])), 4)
        row = {"step": self.step, "epoch": epoch, "n": len(cases), "f1_agent": mean("a"), "f1_copy_c1": mean("c"), "f1_knn": mean("d"),
               "n_rules": len(self.book.rules), "book": self.book.signature()}
        if b is not None:
            row["f1_resample"] = mean("b")
        batch_rec = [case_record(g) for g in groups]
        self.history.extend(batch_rec)
        info = pick_informative(groups, int(sc.n_reflect), float(sc.min_gap), bool(sc.get("balanced_reflect", True)))
        row["n_informative"] = len(info)
        row["n_agent_best"] = sum(1 for g in info if g["regret"] <= 0)
        if not info:
            row["outcome"] = "no_informative_cases"
            return row
        stats = render_stats(batch_rec, self.history, bool(sc.get("history_table", True))) if bool(sc.get("references", True)) else ""
        msgs = [{"role": "system", "content": P.SKILL_OPT_SYSTEM.format(k=int(sc.k_edit), s=max(1, int(sc.get("min_support", 0))),
                                                                       r=float(sc.get("min_support_ratio", 0)) or 1)},
                {"role": "user", "content": P.skill_opt_user(self.book.render_for_optimizer(), self._rejected_text(), render_groups(res, info), stats)}]
        meta = {"cid": f"opt_step{self.step}", "rule_texts": [r.text for r in self.book.rules], "k_edit": int(sc.k_edit),
                "rejected_texts": [e.get("text", "") for rej in self.rejected for e in rej["edits"]], "step": self.step}
        r = self._optimize(msgs, meta)
        edits, analysis = parse_edits(r.data if r.ok else None, int(sc.k_edit))
        row.update(analysis=analysis, proposed=edits)
        if not edits:
            row["outcome"] = "no_edits" if r.ok else f"optimizer_failed:{r.error}"
            return row
        # evidence filter: a rule must cite >= min_support supporting cases, and at least min_support_ratio times
        # as many as the counter-examples it admits (deletions are judged by the gate alone)
        min_sup, ratio, trials, strong = int(sc.get("min_support", 0)), float(sc.get("min_support_ratio", 0)), [], []
        for e in edits:
            s, c = edit_evidence(e, len(info))
            hs, hc = history_evidence(e, stats)
            s, c = s + hs, c + hc
            if str(e.get("op", "")).lower() == "delete" or min_sup <= 0:
                strong.append((e, s))
            elif s < min_sup:
                trials.append({"edit": e, "support": s, "counter": c, "outcome": "weak_support"})
            elif ratio > 0 and s < ratio * c:
                trials.append({"edit": e, "support": s, "counter": c, "outcome": "counter_examples"})
            else:
                strong.append((e, s))
        row["trials"] = trials
        if not strong:
            row["outcome"] = "weak_support"
            return row
        if not bool(sc.get("gate", True)) or str(sc.get("gate_mode", "per_edit")) == "joint":
            return self._joint_update(row, strong)
        # per-edit gate (default): each edit is tested alone against the current book and kept only if it helps
        cur, accepted = self.book, []
        for e, s in strong:
            cand, applied, skipped = cur.apply([e], self.step)
            if not applied:
                trials.append({"edit": e, "support": s, "outcome": skipped[0][1] if skipped else "nothing_applicable"})
                continue
            g = self.gate(cur, cand, applied)
            ok = bool(g["accept"])
            trials.append({"edit": applied[0], "support": s, "gate": g, "outcome": "accepted" if ok else g.get("reason") or "rejected"})
            if ok:
                cur = cand
                r_new = cur.get(applied[0].get("id", ""))
                cf = g.get("confirm") or {}
                if r_new is not None and applied[0]["op"] != "delete":
                    r_new.gain = cf["delta"] if cf.get("tested") else g["delta"]
                accepted.append({**applied[0], "delta": g["delta"], "t": g["t"], "n_tested": g.get("n_tested", 0),
                                 **({"confirm_delta": cf["delta"]} if cf.get("tested") else {})})
            else:
                self.rejected.append({"step": self.step, "edits": applied, "delta": g.get("delta", 0.0), "t": g.get("t", 0.0), "reason": g.get("reason", "")})
        if not accepted:
            row["outcome"] = "rejected" if any("gate" in t for t in trials) else "nothing_applicable"
            return row
        self.book = cur
        row["outcome"] = f"accepted_v{self._new_version(accepted)}"
        return row

    def _new_version(self, edits: list[dict], **extra) -> int:
        from .utils import save_json

        v = len(self.versions) + 1
        save_json(f"{self.run_dir}/versions/book_v{v}.json", self.book.state_dict())
        self.versions.append({"v": v, "step": self.step, "n_rules": len(self.book.rules), "edits": edits, **extra})
        return v

    def _joint_update(self, row: dict, strong: list) -> dict:
        """First-version behaviour, kept for ablations: all edits of the step are applied and gated together
        (gate_mode=joint), or accepted without any check (gate=false)."""
        new_book, applied, skipped = self.book.apply([e for e, _ in strong], self.step)
        row.update(applied=applied, skipped=[{"edit": e, "why": w} for e, w in skipped])
        if not applied:
            row["outcome"] = "nothing_applicable"
            return row
        if bool(self.sc.get("gate", True)):
            g = self.gate(self.book, new_book, applied)
        else:
            g = {"accept": True, "delta": 0.0, "t": 0.0, "n_affected": 0, "reason": "gate_off"}
        row["gate"] = g
        if g["accept"]:
            self.book = new_book
            row["outcome"] = f"accepted_v{self._new_version(applied, delta=g['delta'], t=g['t'])}"
        else:
            self.rejected.append({"step": self.step, "edits": applied, "delta": g.get("delta", 0.0), "t": g.get("t", 0.0), "reason": g.get("reason", "")})
            row["outcome"] = "rejected"
        return row

    def train(self, epochs: int, seed: int = 0, max_steps: int | None = None, on_step=None) -> None:
        from .utils import append_jsonl, stable_shuffle

        B = int(self.sc.batch)
        per_epoch = math.ceil(len(self.practice) / B)
        done = self.step
        for ep in range(epochs):
            order = list(self.practice) if ep == 0 else stable_shuffle(list(self.practice), seed + ep)
            for j in range(per_epoch):
                k = ep * per_epoch + j
                if k < done:
                    continue
                if max_steps is not None and k >= max_steps:
                    return
                snap = copy.deepcopy(self.state_dict())
                try:
                    row = self.run_step(order[j * B: (j + 1) * B], ep)
                except (LLMRequestError, FatalLLMError) as e:
                    self._restore(snap)  # the step is not committed: rule statistics / counters roll back
                    append_jsonl(f"{self.run_dir}/evolve_failures.jsonl", [{"t": round(time.time(), 1), "step": k + 1, "error": str(e), "cost_usd": self.cost()}])
                    LOG.error("第 %d 步有请求多次重试后仍失败，本步未提交，进度停在第 %d 步。网络恢复（或调低 llm.workers）后重新运行同一命令续跑；"
                              "已成功的请求走缓存，不重复收费。", k + 1, k)
                    raise
                row["cost_usd"] = self.cost()
                append_jsonl(self.log_path, [row])
                self.save()
                if on_step:
                    on_step(row)


def evaluate_book(res, cfg, book: SkillBook, cases: list, llm0, voter: KNNVoter | None = None) -> list[dict]:
    sc = cfg.skillbook
    voter = voter or KNNVoter(res, res.select("train"), k=20)
    policy = SkillPolicy(res, voter, int(sc.k_cases), int(sc.scoped_per_case), int(cfg.prescribe.max_herbs), int(cfg.prescribe.vocab_min_freq),
                         int(sc.get("retry_passes", 2)), float(sc.get("retry_wait", 30)))
    return policy.run(book, cases, voter.votes(cases), llm0, desc="eval")
