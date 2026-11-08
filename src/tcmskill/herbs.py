"""Herb lexicon (normalization / tokenization), incompatibility checker, herb info.

Key ideas
---------
* PTM's original preprocessing used *substring* matching against an 811-herb vocabulary,
  which creates artifacts (e.g. "白附子" -> {白附子, 附子}, "罂粟壳" -> {罂粟, 罂粟壳}).
  We re-tokenize with forward maximum matching (FMM) over a lexicon of surface forms.
* The lexicon maps surface forms (vocab names + aliases from kb/herb_aliases.json)
  to *canonical labels*. Synonyms (同物异名) are always merged; processed forms (炮制品)
  are merged when ``merge_processed`` is true.
* The same lexicon normalizes LLM outputs, so ground truth and predictions live in one
  label space.
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Iterable

LOG = logging.getLogger(__name__)

_DOSAGE_UNITS = "钱两分厘铢斤克g升合枚个片只条粒寸尺握撮匙丸字毫ml"
_CN_NUM = "一二三四五六七八九十百千半两零〇"


def fmm(text: str, lexicon: dict[str, str], maxlen: int) -> list[str]:
    """Forward maximum matching. Returns the *mapped* labels in order of appearance."""
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        for L in range(min(maxlen, n - i), 0, -1):
            w = text[i : i + L]
            lab = lexicon.get(w)
            if lab is not None:
                out.append(lab)
                i += L
                break
        else:
            i += 1
    return out


_BRACKETS = re.compile(r"[（(【\[〔][^（(【\[〔）)】\]〕]*[）)】\]〕]")
_TRAILING_DOSE = re.compile(
    r"(各)?([0-9０-９.．]+|[%s]+)\s*(g|G|克|mg|ml|毫升|%s)?\s*半?$" % (_CN_NUM, "|".join(_DOSAGE_UNITS))
)


def strip_brackets(s: str) -> str:
    for _ in range(3):  # nested brackets
        s2 = _BRACKETS.sub(" ", s)
        if s2 == s:
            break
        s = s2
    return s


def basic_clean(name: str) -> str:
    """Remove bracketed notes, whitespace and punctuation (keeps dosage)."""
    s = strip_brackets(str(name or "").strip())
    s = re.sub(r"各?等分|适量|少许|若干", "", s)
    return re.sub(r"[\s，,。.、；;：:\"'“”‘’*＊/／\-—_|]+", "", s)


def strip_dosage(s: str) -> str:
    """Remove a trailing dosage such as '9g', '三钱', '一两半', '各10克' (only at the end)."""
    prev = None
    while prev != s:
        prev = s
        t = _TRAILING_DOSE.sub("", s)
        if t and t != s:
            s = t
    return s


def clean_herb_name(name: str) -> str:
    return strip_dosage(basic_clean(name))


class HerbLexicon:
    def __init__(
        self,
        vocab: list[str],
        alias_path: str,
        merge_processed: bool = True,
        use_aliases: bool = True,
    ):
        self.vocab = list(vocab)
        vocab_set = set(vocab)
        with open(alias_path, encoding="utf-8") as f:
            kb = json.load(f)
        self.prefixes = sorted(kb.get("strip_prefixes", []), key=len, reverse=True)
        self.suffixes = sorted(kb.get("strip_suffixes", []), key=len, reverse=True)

        surface: dict[str, str] = {v: v for v in vocab}
        if use_aliases:
            groups = list(kb.get("synonyms", {}).items())
            proc = list(kb.get("processed", {}).items())
            for canon, aliases in groups + proc:
                if canon not in vocab_set:
                    LOG.warning("alias group canonical '%s' not in vocab; skipped", canon)
                    continue
                is_proc = (canon, aliases) in proc
                for a in aliases:
                    if a in vocab_set and is_proc and not merge_processed:
                        continue  # keep processed vocab entry as its own label
                    surface[a] = canon
            # resolve chains (canonical of canonical) to a fixed point
            for _ in range(5):
                changed = False
                for k, v in list(surface.items()):
                    w = surface.get(v, v)
                    if w != v:
                        surface[k] = w
                        changed = True
                if not changed:
                    break
        self.surface = surface
        self.maxlen = max(len(k) for k in surface)
        self.vocab_only = {v: surface.get(v, v) for v in vocab}  # vocab FMM, but merged labels
        self.vocab_raw = {v: v for v in vocab}  # no merging at all
        self.combos: dict[str, list[str]] = {}
        if use_aliases:
            for k, lst in kb.get("combos", {}).items():
                labs = [surface[x] for x in lst if x in surface]
                if labs:
                    self.combos[k] = labs
        self.labels = sorted(set(surface.values()))
        self.label_set = set(self.labels)

    # ------------------------------------------------------------------ tokenization (GT)
    def tokenize(self, herb_text: str, mode: str = "fmm_alias") -> list[str]:
        """Extract herb labels from a raw prescription text.

        mode = "fmm_alias": FMM over vocab + aliases (default, cleanest)
               "fmm":       FMM over the PTM vocab only (fixes substring artifacts, merges vocab synonyms)
               "ptm":       PTM's original substring matching (no merging) – for comparability
        """
        if mode == "ptm":
            hits = [(herb_text.find(v), v) for v in self.vocab if v and v in herb_text]
            return [v for _, v in sorted(hits)]
        # processing notes in brackets ("厚朴(姜汁炒)") must not produce herbs
        text = strip_brackets(herb_text)
        if mode == "fmm":
            labs = fmm(text, self.vocab_only, max(len(v) for v in self.vocab))
        elif mode == "fmm_alias":
            labs = fmm(text, self.surface, self.maxlen)
        else:
            raise ValueError(mode)
        if not labs:  # very rare: the only herb was inside brackets – fall back to raw text
            labs = fmm(herb_text, self.surface if mode == "fmm_alias" else self.vocab_only, self.maxlen)
        return list(dict.fromkeys(labs))

    # ------------------------------------------------------------------ normalization (LLM output)
    def lookup(self, name: str) -> str | None:
        return self.surface.get(name)

    def normalize(self, name: str) -> list[str]:
        """Map one free-text herb name to canonical labels ([] if out-of-vocabulary)."""
        s0 = basic_clean(name)
        if not s0:
            return []
        for s in dict.fromkeys([s0, strip_dosage(s0)]):
            if not s:
                continue
            labs = self._normalize_clean(s)
            if labs:
                return labs
        return []

    def _normalize_clean(self, s: str) -> list[str]:
        lab = self.surface.get(s)
        if lab:
            return [lab]
        if s in self.combos:
            return list(self.combos[s])
        # strip one processing / origin prefix and/or suffix
        cands = []
        for p in self.prefixes:
            if s.startswith(p) and len(s) - len(p) >= 1:
                cands.append(s[len(p) :])
        for suf in self.suffixes:
            if s.endswith(suf) and len(s) - len(suf) >= 1:
                cands.append(s[: -len(suf)])
                for p in self.prefixes:
                    t = s[: -len(suf)]
                    if t.startswith(p) and len(t) - len(p) >= 1:
                        cands.append(t[len(p) :])
        for c in cands:
            lab = self.surface.get(c)
            if lab:
                return [lab]
        # FMM fallback: accept only if the matched surface covers most of the string
        labs, covered, i = [], 0, 0
        while i < len(s):
            for L in range(min(self.maxlen, len(s) - i), 0, -1):
                w = s[i : i + L]
                if w in self.surface and L >= 2:
                    labs.append(self.surface[w])
                    covered += L
                    i += L
                    break
            else:
                i += 1
        if labs and covered / len(s) >= 0.5:
            return list(dict.fromkeys(labs))
        return []

    def normalize_list(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        out, oov = [], []
        for n in names:
            labs = self.normalize(n)
            if labs:
                out.extend(labs)
            elif str(n).strip():
                oov.append(str(n).strip())
        return list(dict.fromkeys(out)), oov


class IncompatibilityChecker:
    """十八反 / 十九畏 checker operating on canonical labels."""

    def __init__(self, path: str, lexicon: HerbLexicon):
        with open(path, encoding="utf-8") as f:
            kb = json.load(f)
        self.groups: dict[str, set[str]] = {}
        for g, names in kb["groups"].items():
            labs = set()
            for n in names:
                lab = lexicon.lookup(n) or (n if n in lexicon.label_set else None)
                if lab:
                    labs.add(lab)
            self.groups[g] = labs
        self.rules = []
        for r in kb["rules"]:
            for b in r["b"]:
                self.rules.append((r["type"], r["a"], b))

    def violations(self, herbs: Iterable[str]) -> list[dict]:
        hs = set(herbs)
        out, seen = [], set()
        for typ, a, b in self.rules:
            A = hs & self.groups.get(a, set())
            if not A:
                continue
            B = hs & self.groups.get(b, set())
            for x in A:
                for y in B:
                    if x != y and (x, y) not in seen:
                        seen.add((x, y))
                        out.append({"type": typ, "a": x, "b": y})
        return out

    def filter(self, ordered_herbs: list[str]) -> tuple[list[str], list[dict]]:
        """Remove the lower-ranked herb of every violating pair (keeps order)."""
        kept: list[str] = []
        removed: list[dict] = []
        for h in ordered_herbs:
            v = self.violations(kept + [h])
            new = [x for x in v if h in (x["a"], x["b"])]
            if new:
                removed.extend(new)
            else:
                kept.append(h)
        return kept, removed


class HerbInfo:
    """Efficacy categories and short descriptions per canonical herb (from PTM's herb_category)."""

    def __init__(self, info: dict[str, dict]):
        self.info = info
        self.cats = {h: set(v.get("categories", [])) for h, v in info.items()}

    @classmethod
    def load(cls, path: str) -> "HerbInfo":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    def categories(self, herb: str) -> set[str]:
        return self.cats.get(herb, set())

    def describe(self, herb: str, n: int = 60) -> str:
        d = self.info.get(herb, {}).get("desc", "")
        return d[:n]

    def same_category(self, a: str, b: str) -> bool:
        ca, cb = self.categories(a), self.categories(b)
        return bool(ca and cb and (ca & cb))


def build_herb_info(cat_dir: str, lexicon: HerbLexicon, labels: Iterable[str]) -> dict[str, dict]:
    """Read PTM data/herb_category/<category>/<herb>.txt into {canonical: {categories, desc}}."""
    import os

    cats: dict[str, set] = defaultdict(set)
    desc: dict[str, str] = {}
    for c in sorted(os.listdir(cat_dir)):
        cdir = os.path.join(cat_dir, c)
        if not os.path.isdir(cdir):
            continue
        cname = re.sub(r"（中药）", "", c)
        for fn in sorted(os.listdir(cdir)):
            if not fn.endswith(".txt"):
                continue
            name = fn[:-4]
            lab = lexicon.lookup(name) or name
            cats[lab].add(cname)
            if lab not in desc:
                try:
                    with open(os.path.join(cdir, fn), encoding="utf-8") as f:
                        desc[lab] = f.read().strip().replace("\n", " ")[:300]
                except UnicodeDecodeError:
                    pass
    out = {}
    for h in labels:
        out[h] = {"categories": sorted(cats.get(h, [])), "desc": desc.get(h, "")}
    return out
