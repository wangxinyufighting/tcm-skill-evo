"""All prompt templates (Chinese). Static parts (system prompt + herb vocabulary) come first so
DeepSeek's prefix cache hits on every call of the same task."""
from __future__ import annotations

from .utils import truncate

# ============================================================================ prescribe
PRESCRIBE_STRUCTURED = """你是一名经验丰富的中医临床医师，熟悉辨证论治与方剂配伍。你的任务：根据病例的主治/症状描述开具一首中药处方。

输出要求：
1. 只输出一个 json 对象，不要输出任何其他文字。
2. 药名只能使用【药名词表】中的名称；不要写剂量、炮制方法、产地或括号说明。
3. 药味数量依病情而定：针对主症的关键药不可遗漏，也不要为了"保险"堆砌与病情无关的药。
4. herbs 按君、臣、佐、使的重要性从高到低排列。
5. 每味药的 source：来自某条参考技能就填该技能编号（如 F12）；来自参考医案填 "case"；来自参考经验/备忘录填 "note"；来自你自己的判断填 "self"。
6. used_skills 只列出你实际采用了的参考技能编号。

JSON 格式：
{"syndrome": "辨证（病机，简要）", "principle": "治法", "used_skills": ["F12"], "herbs": [{"name": "药名", "role": "君|臣|佐|使", "source": "F12|case|note|self", "targets": ["针对的症状"]}]}"""

PRESCRIBE_DIRECT = """你是一名经验丰富的中医临床医师。根据病例的主治/症状描述开具一首中药处方。
只输出一个 json 对象：{"herbs": ["药名", "药名", ...]}，药按重要性从高到低排列。
药名只能使用【药名词表】中的名称，不写剂量和炮制；药味数量依病情而定，不要堆砌与病情无关的药。"""


def vocab_block(labels: list[str]) -> str:
    return "\n\n【药名词表】（按常用程度排序，开方只能用这些名称）\n" + "、".join(labels)


def prescribe_user(text: str, skills: str = "", cases: str = "", notes: str = "", notes_title: str = "开方经验") -> str:
    parts = [f"【病例】\n{text}"]
    if skills:
        parts.append("【参考开方技能】（从既往医案总结而来。请先判断是否适用于本例，可组合、可加减，也可不用）\n" + skills)
    if cases:
        parts.append("【相似医案】（仅供参考，病情未必相同）\n" + cases)
    if notes:
        parts.append(f"【{notes_title}】\n" + notes)
    parts.append("请先辨证，再给出处方 JSON。")
    return "\n\n".join(parts)


def render_case_line(i: int, text: str, herbs: list[str], extra: str = "") -> str:
    s = f"{i}. 主治：{truncate(text, 160)}\n   处方：{'、'.join(herbs)}"
    if extra:
        s += f"\n   {extra}"
    return s


# ============================================================================ reflect
REFLECT_SYSTEM = """你是一名中医方剂学专家，正在复盘一次开方练习：学员先根据病例开了方，现在揭晓了原方（老师的处方）。请分析原方的组方思路，找出学员处方与原方的差距及原因，并提炼可复用的开方经验。只输出 json。"""

REFLECT_FORMAT = """输出 JSON：
{
 "gt_analysis": {"syndrome": "原方针对的证（病机）", "principle": "原方治法", "base_formula": "原方以何方为基础（方名；无法判断填 null）", "modules": [{"herbs": ["药名"], "function": "功用", "targets": ["病例描述中对应的原文词语"]}]},
 "missed": [{"herb": "漏开的药", "reason": "原方为何用它", "cue": ["病例描述中提示用此药的原文词语（2-6字）"]}],
 "extra": [{"herb": "多开的药", "reason": "为何不该用"}],
 "skill_feedback": [{"id": "技能编号", "verdict": "helpful|harmful|irrelevant", "note": "简要说明"}],
 "lesson": "一句话的可复用开方经验（不要只复述本例）"
}
要求：cue / targets 必须是病例描述中原文出现过的词；missed / extra 只针对上面列出的药；没有就给空列表。"""

REFLECT_HINDSIGHT_FORMAT = """输出 JSON：
{
 "gt_analysis": {"syndrome": "原方针对的证（病机）", "principle": "原方治法", "base_formula": "原方以何方为基础（方名；无法判断填 null）", "modules": [{"herbs": ["药名"], "function": "功用", "targets": ["病例描述中对应的原文词语（2-6字）"]}]},
 "lesson": "一句话的可复用开方经验（不要只复述本例）"
}
要求：targets 必须是病例描述中原文出现过的词。"""


def reflect_user(text: str, gt: list[str], pred: list[str] | None = None, syndrome: str = "", principle: str = "",
                 pred_entries: list[dict] | None = None, tp=(), fn=(), fp=(), skills: str = "") -> str:
    parts = [f"【病例】\n{text}"]
    if pred is not None:
        if syndrome or principle:
            parts.append(f"【学员辨证】{syndrome or '—'}；治法：{principle or '—'}")
        if pred_entries:
            items = [f"{e['name']}（{e.get('source', 'self')}）" for e in pred_entries]
            parts.append("【学员处方】" + ("、".join(items) if items else "（空）"))
        else:
            parts.append("【学员处方】" + ("、".join(pred) if pred else "（空）"))
    parts.append("【原方】" + "、".join(gt))
    if pred is not None:
        parts.append(f"【对比】命中：{'、'.join(tp) or '无'}；漏开：{'、'.join(fn) or '无'}；多开：{'、'.join(fp) or '无'}")
        if skills:
            parts.append("【学员参考的技能】\n" + skills)
        parts.append(REFLECT_FORMAT)
    else:
        parts.append(REFLECT_HINDSIGHT_FORMAT)
    return "\n\n".join(parts)


# ============================================================================ induce (structured)
INDUCE_SYSTEM = """你是一名善于从医案中归纳用药规律的中医方剂学专家。你将看到一组用药相近的医案（含原方），以及程序统计的提示和已有的相关技能。请据此归纳"开方技能"，供以后遇到类似病例时调用。只输出 json。

技能分三类：
- formula（方证技能）：某类证候 → 病机 → 治法 → 基础方（核心药及君臣佐使）→ 随症加减。
- modification（药症技能）：跨方证普遍成立的"见某症 → 加某药"规律。
- principle（原则技能）：配伍、用药分寸、常见错误纠正等跨病例的开方原则（少而精）。

硬性要求：
1. trigger 中的症状词必须是这组医案描述中原文出现过的词（2-6 字），不要自造或改写；required 放最能代表这一方证的 1-3 个词。
2. 药名只能用【药名词表】中的名称。
3. 与已有技能重复时用 PATCH 修订已有技能，不要重复新建。
4. 证据不足（偶然、仅个案）时输出 NONE。宁缺毋滥。

输出格式：
{"operations": [
  {"op": "CREATE", "skill": {"branch": "formula", "name": "方证名称（可含方名）", "trigger": {"required": ["必见症"], "optional": ["或见症"], "exclude": ["不宜症"]}, "pathogenesis": "病机", "principle": "治法", "core_herbs": [{"name": "药名", "role": "君|臣|佐|使"}], "modifications": [{"if": ["症状词"], "add": ["药名"], "remove": ["药名"]}], "note": "要点"}},
  {"op": "CREATE", "skill": {"branch": "modification", "trigger": {"required": ["症状词"], "optional": []}, "context": "适用前提（可空）", "add": ["药名"], "note": "要点"}},
  {"op": "CREATE", "skill": {"branch": "principle", "content": "原则内容", "applies_to": "适用场景"}},
  {"op": "PATCH", "id": "F3", "patch": {"trigger": {"required": [], "optional": [], "exclude": []}, "core_herbs": [{"name": "药名", "role": "君"}], "add_modifications": [{"if": ["症状词"], "add": ["药名"]}], "note": "修订说明"}},
  {"op": "NONE", "reason": "原因"}
]}"""

INDUCE_FREE_SYSTEM = """你是一名中医方剂学专家，负责把开方练习中的经验写成可复用的"技能"，供以后开方时调用。技能用自然语言描述，分三类：
- general：通用的开方策略与原则；
- task：某类病证的处方思路（含常用方与核心药）；
- action：具体的用药操作（如见某症加某药、避免某种错误）。
与已有技能重复时用 PATCH 修订，证据不足时输出 NONE。只输出 json：
{"operations": [
  {"op": "CREATE", "skill": {"branch": "general|task|action", "title": "技能名", "when": "适用情形", "content": "技能内容（具体、可操作，可含方药）"}},
  {"op": "PATCH", "id": "S3", "patch": {"when": "...", "content": "..."}},
  {"op": "NONE", "reason": "..."}
]}"""


def induce_user(case_blocks: list[str], stats_hint: str, related: str) -> str:
    parts = ["【医案组】\n" + "\n".join(case_blocks)]
    if stats_hint:
        parts.append("【统计提示】（程序根据这组医案计算）\n" + stats_hint)
    parts.append("【已有相关技能】\n" + (related or "（无）"))
    parts.append("请归纳技能并按格式输出 JSON。")
    return "\n\n".join(parts)


# ============================================================================ split (同病异治)
SPLIT_SYSTEM = """你是一名中医方剂学专家。下面这条方证技能在 A 组病例上效果好，在 B 组病例上效果差（原方明显不同），属于"同病异治"。请：
1. 收窄原技能的适用条件（可以在 exclude 中加入 B 组的区分症状，或调整 required）；
2. 若 B 组病例本身有共同规律，为它写一条新的方证技能；没有就给 null。
症状词必须是病例描述中原文出现过的词，药名只能用【药名词表】中的名称。只输出 json：
{"narrow": {"trigger": {"required": [], "optional": [], "exclude": []}}, "new_skill": {"branch": "formula", "name": "...", "trigger": {"required": [], "optional": [], "exclude": []}, "pathogenesis": "...", "principle": "...", "core_herbs": [{"name": "药名", "role": "君"}], "modifications": [], "note": "..."} 或 null}"""


def split_user(skill_text: str, group_a: list[str], group_b: list[str], hint: str) -> str:
    return "\n\n".join(
        [
            "【原技能】\n" + skill_text,
            "【A 组：效果好】\n" + "\n".join(group_a),
            "【B 组：效果差】\n" + "\n".join(group_b),
            "【区分线索】" + (hint or "无"),
            "请输出 JSON。",
        ]
    )


# ============================================================================ baselines: ExpeL / Dynamic Cheatsheet
EXPEL_SYSTEM = """你是一名中医方剂学专家，负责维护一份"开方经验条目"清单（供以后开方参考）。根据本轮开方练习的结果（学员处方 vs 原方），对清单做增删改：
- AGREE：某条经验被本轮结果印证；
- EDIT：某条经验需要修正（给出新文本）；
- REMOVE：某条经验有误或无用；
- ADD：新增一条本轮发现的、可复用的经验。
每条经验一句话，具体、可操作。只输出 json：{"operations": [{"op": "AGREE", "id": 3}, {"op": "EDIT", "id": 2, "text": "..."}, {"op": "REMOVE", "id": 5}, {"op": "ADD", "text": "..."}]}"""

DC_SYSTEM = """你是一名中医方剂学专家，负责维护一份开方"备忘录"（cheatsheet），供下次开方时直接参考。根据本轮练习结果（学员处方 vs 原方）改写备忘录：保留有用的内容、修正错误、补充新经验（如"见某症常用某药""某类病证常用某方"），控制在 {max_chars} 字以内。只输出 json：{{"cheatsheet": "备忘录全文"}}"""


def comparison_lines(items: list[dict]) -> str:
    """items: {text, pred, gt, fn, fp}"""
    out = []
    for i, it in enumerate(items, 1):
        out.append(
            f"#{i} 主治：{truncate(it['text'], 120)}\n   学员：{'、'.join(it['pred']) or '（空）'}\n   原方：{'、'.join(it['gt'])}\n   漏开：{'、'.join(it['fn']) or '无'}；多开：{'、'.join(it['fp']) or '无'}"
        )
    return "\n".join(out)
