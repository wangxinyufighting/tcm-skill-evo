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


PRESCRIBE_SELECT_EDIT = """你是一名经验丰富的中医临床医师，擅长"选方加减"：先从相似医案中选出与本病例证候最相符的一首方作为底方，再按本病例的具体症状加减。

步骤：
1. 辨证：概括本病例的病机与治法。
2. 选底方：逐一比较候选医案的主治与本病例是否同证（病机、主症是否一致；字面相似不等于同证）。选出最相符的一首，填其编号（如 "C2"）；若都不对证，填 "none" 并自拟处方。
3. 加减：仅当本病例描述中有明确依据时才加药；底方中与本病例明显无关的药可以减去。加减宜少而精，不要把几首候选方的药合并堆砌。若某个候选医案的主治与本病例几乎相同，原方照用、不加减。

输出要求：
1. 只输出一个 json 对象，不要输出任何其他文字。
2. 药名只能使用【药名词表】中的名称；不写剂量、炮制方法、产地或括号说明。
3. remove 只填底方中要减去的药；add 只填底方之外新增的药，并写出病例描述中的原文依据。
4. herbs 为最终处方的全部药名（= 底方 − remove + add），按君、臣、佐、使排列。

JSON 格式：
{"syndrome": "辨证（病机，简要）", "principle": "治法", "base": "C1", "fit": "底方与本例同证的理由（简要）", "remove": ["药名"], "add": [{"name": "药名", "cue": "原文依据"}], "herbs": ["药名"]}"""


def render_candidate(i: int, text: str, herbs: list[str], sim: float | None = None, extra: str = "") -> str:
    s = f"C{i}" + (f"（sim {sim:.2f}）" if sim is not None else "") + f" 主治：{truncate(text, 160)}\n   处方：{'、'.join(herbs)}"
    if extra:
        s += f"\n   {extra}"
    return s


def select_edit_user(text: str, cands: str, skills: str = "") -> str:
    parts = [f"【病例】\n{text}"]
    parts.append("【候选医案】（按主治描述的字面相似度排序；sim 为相似度 0–1，字面相似不等于同证）\n" + (cands or "（无）"))
    if skills:
        parts.append("【参考开方技能】（从既往医案总结而来，可用于判断证候与加减，也可不用）\n" + skills)
    parts.append("请先辨证、选底方，再加减，输出 JSON。")
    return "\n\n".join(parts)


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


# ============================================================================ skill book (evolve_skills.py)
# The policy prompt is deliberately neutral: no hand-written strategy (when to copy, how many herbs, ...).
# Every strategy the agent follows must come from the evolved 【开方技能】.
SKILL_RX_SYSTEM = """你是一名中医临床医师。根据病例的主治描述，参考相似医案，开具一首中药处方。

输出要求：
1. 只输出一个 json 对象，不要输出任何其他文字。
2. 药名只能使用【药名词表】中的名称；不写剂量、炮制方法、产地或括号说明。
3. 若下面有【开方技能】，请遵循与本例相关的技能；used_rules 列出你实际遵循了的技能编号（如 "R3"），没有就留空。

JSON 格式：
{"analysis": "简要分析（不超过 60 字）", "used_rules": ["R3"], "herbs": ["药名"]}"""


def skill_rx_system(general_rules: str) -> str:
    return SKILL_RX_SYSTEM + ("\n\n【开方技能（通用）】\n" + general_rules if general_rules else "")


def skill_rx_user(text: str, cands: str, scoped_rules: str = "") -> str:
    parts = [f"【病例】\n{text}", "【相似医案】（按主治描述的字面相似度排序；sim 为相似度 0–1）\n" + (cands or "（无）")]
    if scoped_rules:
        parts.append("【开方技能（与本例相关）】\n" + scoped_rules)
    parts.append("请输出 JSON。")
    return "\n\n".join(parts)


SKILL_OPT_SYSTEM = """你是"开方技能"的优化器。一个中医开方智能体按照【当前技能】为病例开方；它看得到病例描述和 5 个相似医案（含相似度），看不到原方。
下面是它在一批练习病例上的作答。每个病例列出了若干种方案（a 智能体作答；如有：b 智能体重新采样、c 照抄最相似医案 C1、d 多个相似医案投票），每种方案都与原方对照打分（F1、精确率 P、召回率 R、味数），并给出了原方。

你的任务：比较同一病例中得分高和得分低的方案（只有一种方案时，比较它与原方的差别），归纳智能体可以改进的、能推广到新病例的开方规律，以"技能编辑"的形式提出修改。

要求：
1. 规律必须可推广：写"什么情况下应该怎么做"（例如如何利用相似医案及其相似度、什么时候照抄、开几味药、如何取舍候选医案中的药、某类病证的用药习惯），不要写只对某一个病例成立的处方。
2. 提出规则前先自查：把规则的适用条件套到下面所有病例上（包括智能体自己得分最高的病例），按规则的做法，哪些病例会变好（支持）、哪些反而变差（反例）？why 里写成 "支持 #2 #5 #9；反例 #4"。支持至少 {s} 个、且至少是反例的 {r:g} 倍，否则不会被检验。只在一两个病例上看到的现象不要提出。
   【统计概况】给出了本批全部病例和累计练习中各方案的平均表现：某种做法整体不如智能体时，不要把它写成普遍规则，只能用在它确实更好的条件下；智能体自己得分最高的病例说明它现有的做法在那里是对的，规则不应改坏这些病例。没有把握时宁可少提或不提编辑。
3. 每条编辑只写一条规则（一个适用条件 + 一个做法），不要把几条规则合并成一条；每条编辑会被单独检验、单独决定是否保留。
4. 每条技能不超过 80 字，是智能体能直接执行的指令；不要空泛的口号。
5. 本轮最多提出 {k} 条编辑：add（新增）、replace（改写已有技能，给出 id）、delete（删除无效或有害的技能，给出 id）。优先改写已有技能，不要新增与已有技能重复的内容。
6. scope：通用技能填 []；只适用于某类病证的技能，填 1–4 个会直接出现在病例描述里的关键词（如 ["产后"]、["惊风","抽搐"]）。
7. 【近期被拒绝的编辑】列出了每条编辑单独在留出病例上的 ΔF1：不要重复提出相同或相近的内容；可以从中判断什么方向无效。
8. 只输出一个 json 对象：
{{"analysis": "高分方案与低分方案的关键差别（不超过 150 字）", "edits": [{{"op": "add", "text": "技能内容", "scope": [], "why": "支持 #2 #5 #9；反例 #4：得分对比"}}, {{"op": "replace", "id": "R3", "text": "新内容", "scope": [], "why": "支持 #1 #4 #7；反例 无"}}, {{"op": "delete", "id": "R5", "why": "#3 #6"}}]}}"""


def skill_opt_user(rules: str, rejected: str, cases: str, stats: str = "") -> str:
    return "\n\n".join([
        "【当前技能】\n" + (rules or "（空：智能体目前没有任何技能）"),
        "【近期被拒绝的编辑】\n" + (rejected or "（无）"),
        *(["【统计概况】\n" + stats] if stats else []),
        "【练习病例】（组内得分差距大的病例：智能体落后的与智能体最好的交替排列）\n" + cases,
        "请输出 JSON。",
    ])


# ============================================================================ offline: name a formula family
ANNOTATE_SYSTEM = """你是一名中医方剂学专家。下面是一组用药相近的古代医案（同一"方族"），以及程序从这些医案统计出的核心药和触发症状词。请为这一方族写出：方证名称（若明显属于某首经典方或其加减，请写出方名）、病机、治法，并标注核心药的君臣佐使。不要增删药物，不要改写症状词。只输出 json：
{"name": "方证名称", "pathogenesis": "病机（简要）", "principle": "治法", "roles": {"药名": "君|臣|佐|使"}, "note": "辨证要点（不超过 40 字）"}"""


def annotate_user(case_blocks: list[str], core: list[str], triggers: str) -> str:
    return "\n\n".join(["【医案】\n" + "\n".join(case_blocks), "【核心药】" + "、".join(core), "【触发症状词】" + (triggers or "无"), "请输出 JSON。"])


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
