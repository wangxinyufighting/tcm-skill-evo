"""Versioned PTM evidence, raw-narrative retrieval, and bounded knowledge views.

Retrieval accepts text, not a target prescription. Observed training cases are an
explicit, separate memory. Retrieval scores express relevance, not clinical truth.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .decision import text_key
from .textindex import TextIndex

VERSION = "ptm-evidence-v1"
VIEWS = ("relation", "syndrome", "case", "integrated")
RELATION_FILES = (
    "herb_symptom_db_knowledge.txt", "herb_symptom_knowledge.txt",
    "symptom_herb_db_knowledge.txt", "symptom_herb_knowledge.txt",
    "symptom_herb_tcm_mesh.txt",
)


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def raw_text(case) -> str:
    text = case.desc_raw
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{case.cid}: knowledge mode requires a nonempty desc_raw")
    return text


@dataclass
class Evidence:
    id: str
    kind: str
    text: str
    herbs: list[str] = field(default_factory=list)
    raw_herbs: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    cid: str | None = None


class KnowledgeBase:
    def __init__(self, evidence: list[Evidence], manifest: dict, symptoms=()):
        self.evidence = {e.id: e for e in evidence}
        self.manifest = manifest
        self.version = digest(manifest)
        self.by_symptom = defaultdict(list)
        self.by_herb = defaultdict(list)
        self.terms = set(symptoms)
        for e in evidence:
            self.terms.update(e.terms)
            if e.kind == "relation":
                for term in e.terms:
                    self.by_symptom[term].append(e.id)
            if e.kind == "herb":
                for herb in e.herbs:
                    self.by_herb[herb].append(e.id)
        # Longest-first matching preserves phrases such as 无汗, rather than 汗.
        terms = sorted((t for t in self.terms if len(t) >= 2), key=lambda t: (-len(t), t))
        self.term_pattern = re.compile("|".join(map(re.escape, terms))) if terms else None
        self.index = self.make_index([(e.id, e.text) for e in evidence if e.kind in {"herb", "syndrome"}])

    @staticmethod
    def make_index(rows):
        if not rows:
            return None
        vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), sublinear_tf=True)
        vec.fit([text for _, text in rows])
        index = TextIndex(vec)
        index.add([eid for eid, _ in rows], [text for _, text in rows])
        return index

    @classmethod
    def load(cls, root: str, lexicon, aliases_path: str | None = None, rules_path: str | None = None):
        root = Path(root)
        if not (root / "herb_category").is_dir():
            raise FileNotFoundError(f"PTM herb_category is missing: {root}")
        evidence, files, warnings, symptoms = {}, {}, [], set()

        def read(path):
            blob = path.read_bytes()
            relative = path.relative_to(root).as_posix()
            files[relative] = hashlib.sha256(blob).hexdigest()
            return blob.decode("utf-8-sig")

        def source(path, line, quote, end_line=None):
            relative = path.relative_to(root).as_posix()
            return {"file": relative, "line": line, "end_line": end_line or line,
                    "sha256": files[relative], "quote": quote, "origin_status": "not_independently_verified"}

        def canonical(name):
            return lexicon.lookup(name) or name

        def add(kind, identity, text, herbs=(), raw_herbs=(), terms=(), sources=(), categories=()):
            eid = "K" + digest([kind, identity])[:20]
            if eid not in evidence:
                evidence[eid] = Evidence(eid, kind, text, list(herbs), list(raw_herbs), list(terms))
            item = evidence[eid]
            item.sources.extend(s for s in sources if s not in item.sources)
            item.categories = sorted(set(item.categories) | set(categories))

        for filename in RELATION_FILES:
            path = root / filename
            if not path.exists():
                warnings.append(f"missing_optional_source:{filename}")
                continue
            for line, row in enumerate(read(path).splitlines(), 1):
                parts = row.split("\t", 1)
                if len(parts) != 2 or not parts[1].strip():
                    continue
                subject = parts[0].strip()
                for obj in parts[1].split():
                    herb, symptom = (subject, obj) if filename.startswith("herb_") else (obj, subject)
                    if not herb or not symptom:
                        continue
                    # Preserve raw herb identity: processing forms are not declared interchangeable.
                    add("relation", [herb, symptom], f"药物：{herb}；相关症状：{symptom}",
                        [canonical(herb)], [herb], [symptom], [source(path, line, row)])

        for path in sorted((root / "herb_category").glob("*/*.txt")):
            text = read(path).strip()
            if text:
                herb = path.stem
                add("herb", [herb, text], text, [canonical(herb)], [herb],
                    sources=[source(path, 1, text, len(text.splitlines()))], categories=[path.parent.name])

        path = root / "symptoms.csv"
        if path.exists():
            original = read(path)
            lines = original.splitlines()
            reader = csv.reader(original.splitlines(keepends=True))
            previous = 0
            for row in reader:
                start, previous = previous + 1, reader.line_num
                if not row or row[0].strip() == "症状":
                    continue
                if len(row) < 3:
                    warnings.append(f"unparsed_symptom_record:{start}")
                    continue
                symptom, syndrome = row[0].strip(), row[1].strip()
                # Unquoted ASCII commas occur in descriptions; retain the remaining fields.
                description = ",".join(row[2:]).strip()
                if not symptom or not syndrome or not description:
                    continue
                quote = "\n".join(lines[start - 1:reader.line_num])
                add("syndrome", [symptom, syndrome, description], f"症状：{symptom}；证候：{syndrome}；表现：{description}",
                    terms=[symptom], sources=[source(path, start, quote, reader.line_num)])
        for filename in ("symptom_category.txt", "symptom_contains.txt"):
            path = root / filename
            if not path.exists():
                continue
            for line, row in enumerate(read(path).splitlines(), 1):
                parts = row.split("\t", 1)
                term = parts[0].strip()
                if term:
                    symptoms.add(term)
                if filename == "symptom_category.txt" and len(parts) == 2 and parts[1].strip():
                    add("syndrome", [term, parts[1]], row, terms=[term], sources=[source(path, line, row)])
        if aliases_path:
            files["@herb_aliases"] = hashlib.sha256(Path(aliases_path).read_bytes()).hexdigest()
        if rules_path:
            files["@incompatibility_rules"] = hashlib.sha256(Path(rules_path).read_bytes()).hexdigest()
        manifest = {"schema": VERSION, "files": dict(sorted(files.items())), "warnings": warnings,
                    "normalization_signature": digest([(e.id, e.herbs, e.terms) for e in sorted(evidence.values(), key=lambda e: e.id)]),
                    "counts": {kind: sum(e.kind == kind for e in evidence.values()) for kind in ("relation", "herb", "syndrome")},
                    "excluded_sources": ["cooccurrence tables", "target prescriptions", "unseen case memory"],
                    "provenance_note": "Repository source locations are verified; upstream medical provenance and independence are not."}
        return cls(sorted(evidence.values(), key=lambda e: e.id), manifest, symptoms)

    def spans(self, text: str) -> list[dict]:
        if self.term_pattern is None:
            return []
        out = []
        for match in self.term_pattern.finditer(text):
            prefix = re.split(r"[，,。；;：:\n]|但|而|伴有|并有", text[:match.start()])[-1]
            negated = bool(re.search(r"(?:否认|未见|不伴|不见|没有|无)(?:明显)?(?:[^，,。；;]{0,6})$", prefix))
            out.append({"term": match.group(), "start": match.start(), "end": match.end(), "negated": negated})
        return out

    def support(self, text: str, herbs: list[str]) -> dict:
        spans = self.spans(text)
        positive = sorted({s["term"] for s in spans if not s["negated"]})
        links = {h: [] for h in herbs}
        covered = set()
        for term in positive:
            for eid in self.by_symptom.get(term, []):
                evidence = self.evidence[eid]
                for h in herbs:
                    if h in evidence.herbs:
                        links[h].append(eid)
                        covered.add(term)
        return {"spans": spans, "positive_terms": positive, "covered_terms": sorted(covered),
                "herb_evidence": {h: sorted(set(ids)) for h, ids in links.items()},
                "unknown_herbs": [h for h, ids in links.items() if not ids],
                "symptom_coverage": len(covered) / len(positive) if positive else None,
                "supported_herb_fraction": sum(bool(ids) for ids in links.values()) / len(herbs) if herbs else None,
                "interpretation": "Positive source support only; missing edges are unknown, not incorrect prescriptions."}


class KnowledgeRetriever:
    def __init__(self, kb: KnowledgeBase, cfg):
        self.kb, self.cfg = kb, cfg
        self.seen = {}
        self.case_index = None
        self.memory_version = digest([])
        self._cache = {}

    def set_seen(self, cases):
        seen = {}
        for case in cases:
            if case.split != "train":
                raise ValueError(f"Only revealed training cases may enter memory: {case.cid}")
            seen[case.cid] = Evidence("C" + digest(case.cid)[:20], "case", raw_text(case),
                                      list(case.herbs), list(case.herbs), cid=case.cid,
                                      sources=[{"file": "revealed_training_memory", "cid": case.cid,
                                                "origin_status": "benchmark_case", "quote": raw_text(case)}])
        self.seen = seen
        self.memory_version = digest([asdict(seen[cid]) for cid in sorted(seen)])
        self.case_index = self.kb.make_index([(e.id, e.text) for e in seen.values()])
        self._cache.clear()

    def _hit(self, e: Evidence, rank: int, route: str, spans: list[dict]) -> dict:
        matched = [s for s in spans if s["term"] in e.text or s["term"] in e.terms]
        limit = int(self.cfg.excerpt_chars)
        positions = [e.text.find(s["term"]) for s in matched if not s["negated"] and s["term"] in e.text]
        start = max(0, min(positions, default=0) - limit // 4)
        excerpt = e.text[start:start + limit]
        return {"id": e.id, "kind": e.kind, "excerpt": excerpt, "excerpt_start": start,
                "excerpt_end": start + len(excerpt), "herbs": e.herbs, "raw_herbs": e.raw_herbs,
                "terms": e.terms, "categories": e.categories, "cid": e.cid,
                "sources": [{k: v for k, v in s.items() if k != "quote"} for s in e.sources],
                "matched_spans": matched, "routes": [route], "rank_score": 1 / (60 + rank),
                "interpretation": "retrieval_match_not_proof"}

    def pool(self, text: str, cid: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty original narrative is required")
        key = digest([text, cid, self.kb.version, self.memory_version, dict(self.cfg)])
        if key in self._cache:
            return self._cache[key]
        spans, hits = self.kb.spans(text), {}

        def add(e, rank, route):
            hit = self._hit(e, rank, route, spans)
            if e.id in hits:
                old = hits[e.id]
                if route not in old["routes"]:
                    old["rank_score"] += hit["rank_score"]
                    old["routes"].append(route)
            else:
                hits[e.id] = hit

        relations = {}
        for span in spans:
            if span["negated"]:
                continue
            for eid in self.kb.by_symptom.get(span["term"], []):
                relations.setdefault(eid, set()).add(span["term"])
        # Specific symptom matches first; a popular herb's degree is not a vote.
        ranked = sorted(relations, key=lambda eid: (-sum(len(t) for t in relations[eid]), eid))
        for rank, eid in enumerate(ranked[:int(self.cfg.relation_k)], 1):
            add(self.kb.evidence[eid], rank, "explicit_symptom_relation")
            if rank <= int(self.cfg.expand_herbs):
                for herb in self.kb.evidence[eid].herbs:
                    for doc in self.kb.by_herb.get(herb, []):
                        add(self.kb.evidence[doc], rank, "relation_to_herb_text")
        if self.kb.index is not None:
            for rank, (eid, _) in enumerate(self.kb.index.query([text], int(self.cfg.text_k))[0], 1):
                add(self.kb.evidence[eid], rank, "raw_narrative_text")
        if self.case_index is not None:
            cases = {e.id: e for e in self.seen.values()}
            excluded = {eid for eid, e in cases.items() if e.cid == cid or text_key(e.text) == text_key(text)}
            for rank, (eid, _) in enumerate(self.case_index.query([text], int(self.cfg.case_k), [excluded])[0], 1):
                add(cases[eid], rank, "revealed_case_text")
        result = {"id": key, "cid": cid, "raw_text": text, "spans": spans, "kb_version": self.kb.version,
                  "memory_version": self.memory_version,
                  "evidence": sorted(hits.values(), key=lambda h: (-h["rank_score"], h["id"]))}
        self._cache[key] = result
        return result

    def packet(self, pool: dict, view: str, variant: int = 0) -> dict:
        patterns = {"relation": ("relation", "relation", "herb", "syndrome"),
                    "syndrome": ("syndrome", "syndrome", "herb", "relation"),
                    "case": ("case", "case", "herb", "relation", "syndrome"),
                    "integrated": ("relation", "herb", "syndrome", "case")}
        if view not in patterns:
            raise ValueError(f"Unknown knowledge view: {view}")
        buckets = {kind: [h for h in pool["evidence"] if h["kind"] == kind] for kind in set(patterns[view])}
        for kind, rows in buckets.items():
            # Rotate only a small top-ranked band, not unrelated documents.
            band = min(len(rows), int(self.cfg.variant_band))
            if band:
                shift = variant % band
                buckets[kind] = rows[shift:band] + rows[:shift] + rows[band:]
        selected, herb_counts, size = [], defaultdict(int), 0
        while any(buckets.values()) and len(selected) < int(self.cfg.max_items):
            for kind in patterns[view]:
                if not buckets[kind] or len(selected) >= int(self.cfg.max_items):
                    continue
                hit = buckets[kind].pop(0)
                if hit["kind"] != "case" and any(herb_counts[h] >= int(self.cfg.per_herb_cap) for h in hit["herbs"]):
                    continue
                serialized = json.dumps(hit, ensure_ascii=False, sort_keys=True)
                if size + len(serialized) + 1 > int(self.cfg.max_chars):
                    continue
                selected.append(hit)
                size += len(serialized) + 1
                if hit["kind"] != "case":
                    for h in hit["herbs"]:
                        herb_counts[h] += 1
        packet = {"schema": VERSION, "pool_id": pool["id"], "view": view, "variant": variant,
                  "kb_version": self.kb.version, "memory_version": self.memory_version,
                  "evidence": selected, "evidence_chars": size, "budget_chars": int(self.cfg.max_chars)}
        packet["id"] = digest(packet)
        return packet

    def packets(self, text: str, cid: str) -> list[dict]:
        pool = self.pool(text, cid)
        return [self.packet(pool, view, variant) for view in self.cfg.views
                for variant in range(int(self.cfg.variants_per_view))]

    @staticmethod
    def render(packet: dict) -> str:
        return "\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in packet["evidence"])
