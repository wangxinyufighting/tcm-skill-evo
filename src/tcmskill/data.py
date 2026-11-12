"""Dataset preparation and loading for PTM (yao8839836/PTM).

What `prepare()` does (see README "数据坑"):
  1. Re-derives PTM's 33,765 preprocessed rows from the 98,334 raw prescriptions and aligns
     them with the official train/test files, so every case gets its full 主治 description
     and the raw herb text (with dosage).
  2. Cleans descriptions (removes 《书名》 citations and source markers to limit leakage).
  3. Re-tokenizes herbs with FMM + alias lexicon (fixes substring artifacts, merges synonyms,
     recovers classical names such as 黄耆/桂心/干地黄).
  4. Splits PTM-train into train/val, clusters cases into categories (κ) with TF-IDF+KMeans.
  5. Flags near-duplicates of train in val/test and computes pattern prevalence buckets.
  6. Writes data/processed/{cases.jsonl, splits.json, meta.json, herb_info.json, report.md}.
"""
from __future__ import annotations

import logging
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
from scipy import sparse

from .herbs import HerbInfo, HerbLexicon, IncompatibilityChecker, build_herb_info, fmm
from .utils import load_json, read_jsonl, save_json, stable_shuffle, write_jsonl

LOG = logging.getLogger(__name__)

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"


# ============================================================================ Case
@dataclass
class Case:
    cid: str
    split: str
    desc: str
    symptoms: list[str]
    herbs: list[str]
    cat: int = -1
    desc_raw: str = ""
    herb_text: str = ""
    herbs_ptm: list[str] = field(default_factory=list)
    flags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(**d)


def case_text(case: Case, input_mode: str = "description") -> str:
    """What the agent sees (and what symbolic trigger matching runs on)."""
    if input_mode == "raw_description":
        if not case.desc_raw or not case.desc_raw.strip():
            raise ValueError(f"{case.cid}: raw_description requires desc_raw")
        return case.desc_raw
    if input_mode == "keywords":
        return "、".join(case.symptoms)
    if input_mode == "both":
        return f"{case.desc}（症状要点：{'、'.join(case.symptoms)}）"
    return case.desc


# ============================================================================ text cleaning
def clean_desc(s: str) -> str:
    s = re.sub(f"[{CIRCLED}]", " ", s)
    s = re.sub(r"(?:引)?《[^》]{0,40}》(?:卷[^：:，。\s]{0,8})?[：:]?", " ", s)  # book citations
    s = re.sub(r"[（(][^（()）]*\d[^（()）]*[）)]", " ", s)  # (1965；12：19) journal refs
    s = re.sub(r"(^|\s)[：:]", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" ：:，,；;")
    return s


# ============================================================================ alignment
def _read_ids(path: str) -> list[tuple[int, ...]]:
    with open(path, encoding="utf-8") as f:
        return [tuple(sorted(map(int, l.split()))) for l in f.read().splitlines()]


def align_ptm(ptm_dir: str) -> tuple[list[dict], list[str], list[str]]:
    """Return aligned rows: [{raw_idx, desc_raw, herb_text, split('train'|'test'), ptm_idx}]."""
    herbs = open(os.path.join(ptm_dir, "herbs_contains.txt"), encoding="utf-8").read().splitlines()
    syms = open(os.path.join(ptm_dir, "symptom_contains.txt"), encoding="utf-8").read().splitlines()
    raw = [l.split("\t") for l in open(os.path.join(ptm_dir, "prescriptions.txt"), encoding="utf-8").read().splitlines()]

    def sub_ids(text: str, vocab: list[str]) -> tuple[int, ...]:
        return tuple(i for i, v in enumerate(vocab) if v and v in text)

    kept = []
    for ri, parts in enumerate(raw):
        if len(parts) != 2:
            continue
        d, h = parts
        s_ids, h_ids = sub_ids(d, syms), sub_ids(h, herbs)
        if s_ids and h_ids:
            kept.append((ri, d, h, (s_ids, h_ids)))
    pre_sig = list(zip(_read_ids(os.path.join(ptm_dir, "pre_symptoms.txt")), _read_ids(os.path.join(ptm_dir, "pre_herbs.txt"))))
    if len(kept) != len(pre_sig) or any(k[3] != p for k, p in zip(kept, pre_sig)):
        raise RuntimeError("raw prescriptions do not reproduce PTM pre_*.txt – unexpected PTM version")
    tr = list(zip(_read_ids(os.path.join(ptm_dir, "pre_symptoms_train.txt")), _read_ids(os.path.join(ptm_dir, "pre_herbs_train.txt"))))
    te = list(zip(_read_ids(os.path.join(ptm_dir, "pre_symptoms_test.txt")), _read_ids(os.path.join(ptm_dir, "pre_herbs_test.txt"))))

    # DP over (#test consumed) to interleave train/test as subsequences of pre order
    n = len(pre_sig)
    back: list[dict[int, int]] = []  # step k: {i_after: i_before}
    states = {0: -1}
    for k in range(n):
        sig = pre_sig[k]
        new: dict[int, int] = {}
        for i in states:
            j = k - i
            if i < len(te) and te[i] == sig and (i + 1) not in new:
                new[i + 1] = i
            if j < len(tr) and tr[j] == sig and i not in new:
                new[i] = i
        if not new:
            raise RuntimeError(f"cannot align train/test at row {k}")
        back.append(new)
        states = new
    if len(te) not in states:
        raise RuntimeError("alignment did not consume all test rows")
    labels = [""] * n
    i = len(te)
    for k in range(n - 1, -1, -1):
        prev = back[k][i]
        labels[k] = "test" if prev != i else "train"
        i = prev
    rows = []
    c = Counter()
    for k, (ri, d, h, _) in enumerate(kept):
        sp = labels[k]
        rows.append({"raw_idx": ri, "desc_raw": d, "herb_text": h, "split": sp, "ptm_idx": c[sp]})
        c[sp] += 1
    assert c["train"] == len(tr) and c["test"] == len(te)
    return rows, herbs, syms


# ============================================================================ prepare
def prepare(cfg, ptm_dir: str, out_dir: str, kb_dir: str) -> dict:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    dcfg = cfg.data
    rows, ptm_herbs, ptm_syms = align_ptm(ptm_dir)
    LOG.info("aligned %d rows (%d train / %d test)", len(rows), sum(r["split"] == "train" for r in rows), sum(r["split"] == "test" for r in rows))
    lex = HerbLexicon(
        ptm_herbs,
        os.path.join(kb_dir, "herb_aliases.json"),
        merge_processed=dcfg.merge_processed,
        use_aliases=dcfg.gt_mode == "fmm_alias",
    )
    sym_lex = {s: s for s in ptm_syms}
    sym_max = max(map(len, ptm_syms))

    cases: list[Case] = []
    n_cit = 0
    for r in rows:
        desc = clean_desc(r["desc_raw"])
        n_cit += desc != r["desc_raw"].strip()
        syms = list(dict.fromkeys(fmm(desc, sym_lex, sym_max))) or list(dict.fromkeys(fmm(r["desc_raw"], sym_lex, sym_max)))
        herbs = lex.tokenize(r["herb_text"], dcfg.gt_mode)
        herbs_ptm = lex.tokenize(r["herb_text"], "ptm")
        pre = "tr" if r["split"] == "train" else "te"
        cases.append(
            Case(
                cid=f"{pre}{r['ptm_idx']:05d}",
                split=r["split"],
                desc=desc or r["desc_raw"],
                symptoms=syms,
                herbs=herbs,
                desc_raw=r["desc_raw"],
                herb_text=r["herb_text"],
                herbs_ptm=herbs_ptm,
            )
        )
    # ---------------- splits
    train_ids = [c.cid for c in cases if c.split == "train"]
    test_ids = [c.cid for c in cases if c.split == "test"]
    shuffled = stable_shuffle(train_ids, dcfg.seed)
    val_ids = set(shuffled[: dcfg.val_size])
    by_id = {c.cid: c for c in cases}
    for cid in val_ids:
        by_id[cid].split = "val"
    train_ids = [c for c in train_ids if c not in val_ids]
    val_ids_l = [c for c in shuffled[: dcfg.val_size]]

    # ---------------- categories κ (TF-IDF char n-grams + KMeans, fit on train only)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    txt = lambda c: case_text(c, dcfg.input_mode)
    Xtr = vec.fit_transform([txt(by_id[c]) for c in train_ids])
    km = MiniBatchKMeans(n_clusters=dcfg.n_categories, random_state=dcfg.seed, n_init=3, batch_size=4096)
    km.fit(Xtr)
    for cid, lab in zip(train_ids, km.labels_):
        by_id[cid].cat = int(lab)
    other = [c for c in by_id if by_id[c].split != "train"]
    for cid, lab in zip(other, km.predict(vec.transform([txt(by_id[c]) for c in other]))):
        by_id[cid].cat = int(lab)

    # ---------------- label space (train GT)
    herb_freq = Counter(h for cid in train_ids for h in by_id[cid].herbs)
    labels = [h for h, _ in herb_freq.most_common()]

    # ---------------- near-duplicate & prevalence analysis for val/test
    lab_index = {h: i for i, h in enumerate(lex.labels)}

    def herb_matrix(ids):
        r, c = [], []
        for i, cid in enumerate(ids):
            for h in by_id[cid].herbs:
                r.append(i)
                c.append(lab_index[h])
        return sparse.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(ids), len(lex.labels)))

    Htr = herb_matrix(train_ids)
    tr_len = np.asarray(Htr.sum(1)).ravel()
    tr_desc = {}
    for cid in train_ids:
        tr_desc.setdefault(by_id[cid].desc, []).append(cid)
    for split_ids in (val_ids_l, test_ids):
        Q = herb_matrix(split_ids)
        q_len = np.asarray(Q.sum(1)).ravel()
        Xq = vec.transform([txt(by_id[c]) for c in split_ids])
        for start in range(0, len(split_ids), 500):
            sl = slice(start, start + 500)
            inter = (Q[sl] @ Htr.T).toarray()
            jac = inter / np.maximum(q_len[sl, None] + tr_len[None, :] - inter, 1e-9)
            sims = (Xq[sl] @ Xtr.T).toarray()
            for row in range(inter.shape[0]):
                cid = split_ids[start + row]
                nn = np.argpartition(-sims[row], 20)[:20]
                nn = nn[np.argsort(-sims[row][nn])]
                near = any(sims[row][j] >= 0.85 and jac[row][j] >= 0.7 for j in nn)
                same_desc = tr_desc.get(by_id[cid].desc, [])
                exact = any(set(by_id[t].herbs) == set(by_id[cid].herbs) for t in same_desc)
                prev = int((jac[row] >= 0.6).sum())
                by_id[cid].flags = {
                    "near_dup": bool(near or exact),
                    "exact_dup": bool(exact),
                    "desc_dup": bool(same_desc),
                    "max_desc_sim": round(float(sims[row][nn[0]]), 4),
                    "nn_herb_jacc": round(float(jac[row][nn[0]]), 4),
                    "max_herb_jacc": round(float(jac[row].max()), 4),
                    "prevalence": prev,
                    "prev_bucket": "rare" if prev == 0 else ("mid" if prev < 10 else "common"),
                }

    # ---------------- knowledge
    herb_info = build_herb_info(os.path.join(ptm_dir, "herb_category"), lex, lex.labels)
    inc = IncompatibilityChecker(os.path.join(kb_dir, "incompatibility.json"), lex)

    # ---------------- write
    os.makedirs(out_dir, exist_ok=True)
    write_jsonl(os.path.join(out_dir, "cases.jsonl"), (c.to_dict() for c in cases))
    test_order = stable_shuffle(test_ids, dcfg.seed + 1)
    splits = {
        "train": train_ids,
        "val": val_ids_l,
        "test": test_ids,
        "test_order": test_order,  # fixed random order: --n_test N takes the first N
        "stream_random": stable_shuffle(train_ids, dcfg.seed + 2),
    }
    splits["stream_category"] = sorted(splits["stream_random"], key=lambda c: by_id[c].cat)  # stable sort keeps random order inside a category
    save_json(os.path.join(out_dir, "splits.json"), splits, indent=None)
    save_json(os.path.join(out_dir, "herb_info.json"), herb_info)
    meta = {
        "gt_mode": dcfg.gt_mode,
        "merge_processed": dcfg.merge_processed,
        "input_mode_for_categories": dcfg.input_mode,
        "ptm_vocab": ptm_herbs,
        "ptm_symptoms": ptm_syms,
        "herb_freq": [[h, herb_freq[h]] for h in labels],
        "n_categories": dcfg.n_categories,
        "seed": dcfg.seed,
    }
    save_json(os.path.join(out_dir, "meta.json"), meta, indent=None)
    report = build_report(cases, by_id, train_ids, val_ids_l, test_ids, lex, inc, herb_info, herb_freq, n_cit)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report["markdown"])
    save_json(os.path.join(out_dir, "report.json"), report["stats"])
    return report["stats"]


def build_report(cases, by_id, train_ids, val_ids, test_ids, lex, inc, herb_info, herb_freq, n_cit) -> dict:
    def mean(x):
        x = list(x)
        return sum(x) / max(len(x), 1)

    tr = [by_id[c] for c in train_ids]
    te = [by_id[c] for c in test_ids]
    # compare in the same label space (PTM names mapped through the lexicon) so that pure renames don't count
    ptm_mapped = lambda c: {lex.surface.get(h, h) for h in c.herbs_ptm}
    n_diff = sum(set(c.herbs) != ptm_mapped(c) for c in cases)
    n_added = sum(len(set(c.herbs) - ptm_mapped(c)) for c in cases)
    n_removed = sum(len(ptm_mapped(c) - set(c.herbs)) for c in cases)
    sym_len = Counter(min(len(c.symptoms), 5) for c in cases)
    gt_viol = sum(1 for c in cases if inc.violations(c.herbs))
    covered = sum(cnt for h, cnt in herb_freq.items() if herb_info.get(h, {}).get("categories"))
    st = {
        "n_cases": len(cases),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "n_test": len(test_ids),
        "n_labels_total": len(lex.labels),
        "n_labels_train": len(herb_freq),
        "herb_len_mean_gt": round(mean(len(c.herbs) for c in cases), 3),
        "herb_len_mean_ptm": round(mean(len(c.herbs_ptm) for c in cases), 3),
        "cases_gt_differs_from_ptm": n_diff,
        "herbs_added_vs_ptm": n_added,
        "herbs_removed_vs_ptm": n_removed,
        "desc_len_mean": round(mean(len(c.desc) for c in cases), 1),
        "symptoms_per_case_mean": round(mean(len(c.symptoms) for c in cases), 3),
        "symptoms_per_case_hist": {str(k) if k < 5 else "5+": v for k, v in sorted(sym_len.items())},
        "descriptions_with_citations_removed": n_cit,
        "gt_with_incompatible_pair": gt_viol,
        "herb_category_coverage_weighted": round(covered / max(sum(herb_freq.values()), 1), 4),
        "test_near_dup": sum(c.flags.get("near_dup", False) for c in te),
        "test_exact_dup": sum(c.flags.get("exact_dup", False) for c in te),
        "test_desc_dup": sum(c.flags.get("desc_dup", False) for c in te),
        "test_prev_buckets": dict(Counter(c.flags.get("prev_bucket") for c in te)),
        "test_nn1_herb_jacc_mean": round(mean(c.flags.get("nn_herb_jacc", 0) for c in te), 4),
        "category_sizes_train": dict(sorted(Counter(c.cat for c in tr).items())),
    }
    top = ", ".join(f"{h}({n})" for h, n in herb_freq.most_common(30))
    md = [
        "# 数据报告（自动生成）",
        "",
        "| 项目 | 数值 |",
        "|---|---|",
    ]
    names = {
        "n_cases": "病例总数",
        "n_train": "训练集",
        "n_val": "验证集（从 PTM 训练集划出）",
        "n_test": "测试集（PTM 官方）",
        "n_labels_total": "药物标签总数（归一化后）",
        "n_labels_train": "训练集出现的药物标签数",
        "herb_len_mean_gt": "GT 平均味数（本项目切分）",
        "herb_len_mean_ptm": "GT 平均味数（PTM 原始子串匹配）",
        "cases_gt_differs_from_ptm": "GT 与 PTM 原始药物集合不同的病例数（同一标签空间下比较）",
        "herbs_added_vs_ptm": "相对 PTM 新增的药物次数（别名恢复，如 黄耆→黄芪）",
        "herbs_removed_vs_ptm": "相对 PTM 删除的药物次数（子串伪影，如 白附子→附子）",
        "desc_len_mean": "主治描述平均字数",
        "symptoms_per_case_mean": "每例症状关键词数（PTM 词表）",
        "descriptions_with_citations_removed": "去除了书名引文的描述数",
        "gt_with_incompatible_pair": "GT 本身含十八反/十九畏配伍的病例数",
        "herb_category_coverage_weighted": "功效分类覆盖率（按频次加权）",
        "test_near_dup": "测试集中与训练集近重复的病例数",
        "test_exact_dup": "测试集中与训练集完全重复（描述+药物）",
        "test_desc_dup": "测试集中描述与训练集完全相同的病例数",
        "test_nn1_herb_jacc_mean": "测试集：描述最近邻训练病例的药物 Jaccard 均值",
    }
    for k, v in st.items():
        if k in names:
            md.append(f"| {names[k]} | {v} |")
    md += [
        "",
        f"症状关键词数分布：{st['symptoms_per_case_hist']}",
        "",
        f"测试集模式常见度分桶（训练集中药物 Jaccard≥0.6 的病例数：rare=0, mid=1-9, common≥10）：{st['test_prev_buckets']}",
        "",
        f"训练集高频药物前 30：{top}",
        "",
        f"训练集类别（κ）大小：{st['category_sizes_train']}",
        "",
        "## 解读提示",
        "- 症状关键词极稀疏（多数病例只有 1 个），因此默认把清洗后的完整主治描述作为输入；关键词只作为辅助特征。",
        "- 近重复 / 常见模式的测试病例对检索类方法（kNN、病例 RAG）非常有利，务必同时报告去重子集（near_dup=False）的结果。",
        "- GT 自身含配伍禁忌的比例作为安全指标的参照：模型违规率应与之对比解读。",
    ]
    return {"stats": st, "markdown": "\n".join(md) + "\n"}


# ============================================================================ runtime resources
class Resources:
    """Everything a method needs at run time (loaded from data/processed)."""

    def __init__(self, cfg):
        pdir = cfg.paths.processed_dir
        if not os.path.exists(os.path.join(pdir, "cases.jsonl")):
            raise FileNotFoundError(f"{pdir}/cases.jsonl not found – run scripts/prepare_data.py first")
        self.cfg = cfg
        self.meta = load_json(os.path.join(pdir, "meta.json"))
        self.cases: dict[str, Case] = {d["cid"]: Case.from_dict(d) for d in read_jsonl(os.path.join(pdir, "cases.jsonl"))}
        self.splits: dict[str, list[str]] = load_json(os.path.join(pdir, "splits.json"))
        self.lexicon = HerbLexicon(
            self.meta["ptm_vocab"],
            os.path.join(cfg.paths.kb_dir, "herb_aliases.json"),
            merge_processed=self.meta["merge_processed"],
            use_aliases=self.meta["gt_mode"] == "fmm_alias",
        )
        self.herb_freq: dict[str, int] = {h: n for h, n in self.meta["herb_freq"]}
        self.labels: list[str] = [h for h, _ in self.meta["herb_freq"]]  # by train frequency
        self.n_train = len(self.splits["train"])
        self.herb_info = HerbInfo.load(os.path.join(pdir, "herb_info.json"))
        self.incompat = IncompatibilityChecker(os.path.join(cfg.paths.kb_dir, "incompatibility.json"), self.lexicon)
        self.input_mode = cfg.data.input_mode
        lens = sorted(len(self.cases[c].herbs) for c in self.splits["train"])
        self.median_len = lens[len(lens) // 2]
        self._vectorizer = None

    def text(self, case: Case) -> str:
        return case_text(case, self.input_mode)

    def get(self, cids: Iterable[str]) -> list[Case]:
        return [self.cases[c] for c in cids]

    def herb_prior(self, h: str) -> float:
        return self.herb_freq.get(h, 0) / max(self.n_train, 1)

    @property
    def vectorizer(self):
        """Char n-gram TF-IDF fitted on train texts (shared by kNN / case retrieval / skill text sim)."""
        if self._vectorizer is None:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            vec.fit([self.text(self.cases[c]) for c in self.splits["train"]])
            self._vectorizer = vec
        return self._vectorizer

    def select(self, split: str, n: int | None = None, order: str | None = None) -> list[str]:
        """Deterministic subsets. split in {train,val,test}; order in {None,random,category}."""
        if split == "test":
            ids = self.splits["test_order"]
        elif split == "val":
            ids = self.splits["val"]
        elif split == "train":
            ids = self.splits["stream_category" if order == "category" else "stream_random"]
            if order == "category" and n:
                # keep category order but subsample uniformly so every category is represented
                keep = set(self.splits["stream_random"][:n])
                ids = [c for c in ids if c in keep]
        else:
            raise ValueError(split)
        return list(ids[:n]) if n else list(ids)
