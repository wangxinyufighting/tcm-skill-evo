# 知识条件化的开方技能进化

> 历史说明：本页记录 v5 的观察式补证流程，不是当前默认训练循环。当前命令和条件见 [v7 多方案对比与技能迭代](knowledge_exploration_v7.md)。`learning_mode=observe` 仅保留观察式消融，仍使用新的审核实现，不等于精确复现旧 v5。

本入口实现“原始主治全文 + 多知识包 → 药方及生成时的依据 → 多维评分 → 跨批次对比经验 → 假设库 → 待验证库 → 新留出病例门控 → 正式技能库”。
LLM 参数不更新，只更新技能库。旧的 `scripts/evolve_skills.py`、既有实验目录和既有指标输出不改写。
本页所述协议是 v5；以检索召回正式技能，再由 LLM 按完整原文判断适用性，不再把所有必要条件当作逐字短语触发器。
延续 v4 的程序性/医学性证据要求及必要事实、操作步骤、可选分支、排除项区分；不因当前病例缺少必要事实而丢弃整个技能假设。
全部判断条件、默认阈值、实现位置和未实现边界见 [判断条件清单](knowledge_judgment_rules.md)。

## 三类技能库

| 库 | 内容与转移条件 | 普通开方/评测是否可用 |
| --- | --- | --- |
| 假设库 | 灵感、暂缺适用病例、独立支持不足，以及待修订的定义/失败方案；保留全文、来源、版本及判断历史 | 否 |
| 待验证库 | 定义审核通过，默认至少 3 个独立原文描述组同时满足适用性和候选对比支持；等待新留出组门控 | 否；只在受控门控的试验侧使用 |
| 正式技能库 | 通过初筛及启用时的独立复核；仍须逐病例检查适用性 | 是 |

- `origin_cases` 是产生想法的来源，不等于 `supported_cases`。优化器只需看到 1 组有区分度的经验即可提出假设，允许支持列表为空。
- 每步自动从当前已完成生成/评分的练习和已提交经验池中，为可检查的假设检索相关案例。必要事实未知时只记 `uncertain`，不是反证，更不会把这些事实删掉来强行适用。
- 适用性成立后，还需 `knowledge_skill_support` 判断合法组内优劣对的具体策略差异是否支持技能。仅有高 F1 或病例适用不够；保存 `supports/opposes/unclear`、候选对、原文引用和理由。这是观察性支持，不证明因果。
- 默认 `hypothesis_review_budget=12`：每步至多检查 12 个“假设-病例”组合，每组至多 1 次适用性和 1 次对比审核。同一原文描述组只累计一次；不能重复抽取或反复评判来凑支持。已存观察随断点持久化，不受 256 组经验池淘汰影响。
- 默认 `pending_gate_budget=1`：每步从待验证库取至多 1 条做门控/预筛。覆盖不足继续保留；下步自动接着未查过的原文组扫描，每次新增检查不超过 `gate_candidate_limit`，不必让优化器重复提出它。正式技能或操作定义改变后重新建立扫描上下文。
- 格式错误、引用无效、定义混用仍入假设库，但标记 `needs_revision`，不能自动执行。真正行为门控失败也退回假设库，保留分数与原因；同一定义不能靠重复提案重抽门控碰运气。
- 修订用 `revises: "H编号"` 关联新定义。实质字段变化会建立新假设，重新积累支持；旧记录保留并归档。仅换支持案例、引用或重复原提案不会重置失败锁。去重按规范化定义的确定性签名，不声称识别医学语义等价的改写。
- `opposes` 记录反向观察，不自动把所有支持作废，也不是临床错误标签；决定是否正式采用的仍是独立行为门控。删除正式技能是编辑操作，可直接进入待验证队列，但仍须通过门控。
- **只把练习当支持证据，绝不把门控、val、test 当支持病例。** 假设、待验证项的摘要只给优化器，不进入普通处方提示词。

`complete=true` 只表示预定练习流已处理完，不表示所有假设已经得到验证。原先只有 `rules=0` 的返回值，现在还能看 `hypotheses`、`pending_skills`、`formal_skills`、`hypotheses_needing_revision`；归档历史单列 `archived_hypotheses`，也计入假设库总条数。
每个被处理的提案都保留其版本；整体优化器响应不是合法 JSON、没有 edits 列表或超出 k_edit 的额外项，仍只保留原始响应日志，不据此编造技能。

## v5 小规模运行

旧 v1–v4 运行不可跨协议续跑；复用 `knowledge_test_4` 的处方与生成时理由，不复用其旧提案、正式技能或适用性判断：

```bash
python scripts/evolve_knowledge.py train --llm openai_compatible --run_name knowledge_v5_pilot64 \
  --reuse_run runs/knowledge_test_4 \
  --set knowledge.n_practice=64 --set knowledge.batch=8 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 \
  --set knowledge.min_support=3 --max_steps 4
```

先处理约 32 例；初始 16 例优先回放。保持配置不变、增加累计 max_steps 可继续，三类库一起恢复。这里不降低门控收益阈值，也不保证必有技能通过。
应提前规划 n_practice，再用 max_steps 控制阶段性运行。当前不支持在已完成运行中改 n_practice 扩流，也不跨运行导入假设库；新 run 的 --reuse_run 仅回放候选。

## 快速使用

使用安装了本项目依赖的 Python 3.10+ 环境。下面的 mock 命令不调用模型 API；不代表处方质量或方法收益。

```bash
# 只检索原文并导出知识包，不调用 LLM，不读取该病例的原方作为查询条件。
python scripts/evolve_knowledge.py inspect \
  --text '咳嗽，口渴，无发热。' --out runs/knowledge_inspect_v1

# 小规模模拟：四种知识视角、结构化理由、评分、编辑和门控。
python scripts/evolve_knowledge.py train --llm mock --run_name knowledge_mock_v1 \
  --set knowledge.n_practice=16 --set knowledge.batch=4 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 \
  --set knowledge.min_support=1 --max_steps 2

# 原命令续跑；去掉 max_steps 后处理整个练习分区。
python scripts/evolve_knowledge.py train --llm mock --run_name knowledge_mock_v1 \
  --set knowledge.n_practice=16 --set knowledge.batch=4 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 \
  --set knowledge.min_support=1

# 固定记忆、知识库、知识视角，只比较是否提供技能。
python scripts/evolve_knowledge.py eval --llm mock \
  --run_dir runs/knowledge_mock_v1 --split val --n 30 --out runs/knowledge_mock_eval
python scripts/evolve_knowledge.py eval --llm mock --empty \
  --run_dir runs/knowledge_mock_v1 --split val --n 30 --out runs/knowledge_mock_empty
```

真实调用需要显式使用 `--llm deepseek` 或 `--llm openai_compatible`，并设置端点、模型和 API 环境变量。
默认模型设置沿用仓库当前配置，不保证端点可访问。首次真实运行建议保持小样本和 `--max_steps 1`。
`knowledge.min_support=1` 仅适合流程冒烟，正式实验默认至少 3 个独立描述组。

### 复用已有候选，先做小实验

已有 `knowledge_test_1` / `knowledge_test_2` 的候选不必重生成；当前目录已移至 `runs/Deprecated/`。必须使用新运行名，不可直接续跑旧协议。
先用只读来源、零模型调用的命令验证导入，保留原实验的 practice 数量、batch 和 seed：

```bash
python scripts/evolve_knowledge.py train --llm mock --run_name knowledge_replay_audit_v5 \
  --reuse_run runs/Deprecated/knowledge_test_2 --reuse_run runs/Deprecated/knowledge_test_1 \
  --replay_only --collect_only \
  --set knowledge.n_practice=16 --set knowledge.batch=4 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 --max_steps 2
```

再用真实模型归纳、判断适用性及门控。以下命令优先复用同样的历史候选；后续遇到未覆盖病例时才生成新候选。
`gate_candidate_limit=24` 约束每次预筛新增检查的候选描述组；未完成的覆盖扫描在后续步骤继续，避免单步扫描全训练集。

```bash
export DEEPSEEK_API_KEY=magpie
python scripts/evolve_knowledge.py train --llm openai_compatible --run_name knowledge_semantic_v5 \
  --reuse_run runs/Deprecated/knowledge_test_2 --reuse_run runs/Deprecated/knowledge_test_1 \
  --set knowledge.n_practice=16 --set knowledge.batch=4 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 \
  --set knowledge.gate_candidate_limit=24 --set knowledge.min_support=3 --max_steps 2
```

使用当前配置的端点和模型。只将 `--max_steps 2` 改为 `--max_steps 4` 可在相同配置下继续。
4+4 门控只是流程小试，不足以支持稳定收益结论；正式实验应扩大样本。没有技能通过门控仍是合法结果。

- `--reuse_run` 可重复；只导入来源 `state.json` 已提交的练习候选，不导入旧技能、门控处方或未提交步骤。
- 数据签名、知识库版本、完整原文和原方必须匹配。同 cid 在多个来源出现时固定取首个来源的候选，不挑最高分。
- 历史候选按当前裁判重新评分，标记 `provenance.kind=replay`，保存来源协议、步骤签名及来源模型。
  它们是历史策略的经验，不能用来声称当前技能库的处方性能。
- 来源已见练习病例及已分配门控病例，连同上游 replay 曝光记录，均从新门控池按描述组排除。
- 候选引用的训练医案须在新流中也已揭晓；未来医案或自例不能通过回放泄漏。batch/seed 改动可能使回放不兼容。
- `--replay_only` 遇到不兼容或缺失候选即报错；只禁止新增练习处方，不禁止优化器、适用性或门控调用。
- `--collect_only` 只收集经验，不提出/验证技能。只有同时使用这两个开关且候选完整兼容时，整个运行才没有 LLM 调用。
- 导入内容冻结为 `replay_archive.json` 并绑定协议；续跑从快照恢复，不依赖源目录仍在原位置。
  收集模式和训练模式不能在同一运行内切换；需要另建 run，以收集运行作为 `--reuse_run` 来源。

### 本地 WorkBuddy Hy4 的输出预算

本地代理 `http://127.0.0.1:3425/v1/models` 在本次排查时为 `workbuddy-ai/hy4free` 宣告
`reasoning_levels=[high]`、`max_output_tokens=64000`。客户端 `thinking=false` 在兼容模式下不发送关闭思考参数，
不能据此认定后端关闭了思考；知识入口的 agent 虽使用这个客户端模式，实际仍可能产生 reasoning。

若收到 `502 ... empty reply ... whole output spent on reasoning`，这是端点明确报告额度耗尽，不是 API Key 错误。
建议先把 `llm.max_tokens`、`llm.thinking_max_tokens` 与 `skillbook.optimizer_max_tokens` 都限制在 64000，
串行调用，不发送 JSON-mode 扩展参数。64000 是上限，不代表每次必定消耗这么多 token。
`thinking_max_tokens` 并非额外的思考额度：客户端开启 thinking 时，实际请求取它与本次 max_tokens 的较大值，
因此不能只调整后者而遗留一个超过上限的前者。160000 或 640000 都超过该端点本次宣告的最大输出。
对此特定错误，客户端记录实际预算和原始错误链后立即停止，避免反复用同一额度请求；普通临时 502 仍按原规则重试。
配置或实现变化后，使用新的运行目录，不修改旧协议来强行续跑。此处以代理声明为依据，不保证远端模型别名和能力长期不变。

2026-11-12 的单次完整病例诊断使用 integrated 知识包、空技能库、64000 输出上限和 600 秒超时，
未使用原方：217.5 秒后返回完整 JSON，`finish_reason=stop`，输出 16617 token，其中 reasoning 15540 token。
该例的思考量和耗时都超过了此前的 8192 token / 180 秒设置；这只是一次连通性与输出格式验证，不是正式训练或处方质量验证。

下面只验证一个练习病例的四种知识视角，不够归纳正式技能。显式覆盖预算以免本地配置的试验值影响请求。
长思考可能超过原有 180 秒；600 秒只是诊断等待时间，不保证后端成功。首次验证关闭传输和病例层重试，避免故障时重复等待。

```bash
export DEEPSEEK_API_KEY=magpie
python scripts/evolve_knowledge.py train --llm openai_compatible --run_name knowledge_hy4_probe_v1 \
  --set llm.model=workbuddy-ai/hy4free --set llm.base_url=http://127.0.0.1:3425/v1 \
  --set llm.max_tokens=64000 --set llm.thinking_max_tokens=64000 \
  --set skillbook.optimizer_max_tokens=64000 --set llm.timeout=600 \
  --set llm.workers=1 --set llm.json_mode=false --set llm.max_retries=0 \
  --set skillbook.retry_passes=0 --set knowledge.n_practice=4 --set knowledge.batch=1 --max_steps 1
```

## 数据与检索

- 新入口强制使用 `desc_raw`，缺失则报错，不回退到关键词或清洗后文本。原始病例全文不截断。
- 药症关系来自五份显式关系表；反向重复关系合并来源，不把重复文件当作独立支持。
- 药物功效与主治直接读取 `herb_category/*/*.txt` 完整原文，不使用截断后的 `herb_info.json` 代替知识库。
- 证候材料来自 `symptoms.csv` 和 `symptom_category.txt`。引用保留源文件、行号、文件 SHA256，快照保留原始引用文本。
- 原始药名、炮制形式和规范名称同时保存；规范名称命中不表示炮制形式可互换。
- 文件路径是仓库来源，不冒充已核实的教材出处。上游医学来源及其与数据集的独立性仍标为未独立核验。
- 不读取 `*_cooccur*` 表，不从目标处方构建查询，不依据原方补造症状或证型。

召回包含三路：完整原文字符 TF-IDF、带原文位置的症状关系查表、已揭晓训练医案全文检索。
仅做“症状 → 药物 → 药物原文”的有限扩展。证候检索命中是相关线索，不是确诊或新的证候药物关系。
知识索引独立拟合；医案索引只在当前已见病例上拟合。最长词匹配和否定前缀只是可审计启发式，不是完整古文语义解析。
未提到的症状视为未知，明确否定的症状不用于正向药症关系扩展。原文仍完整交给模型。

四种知识包：药症型 `relation`、证候型 `syndrome`、医案型 `case`、综合型 `integrated`。
包内不同类型使用不同配额顺序；每包有相同的证据字符预算和条目上限。预算不包含完整病例原文和固定提示。
`variants_per_view=2` 可扩展为八个候选，只轮换小范围的高相关证据，不靠提高温度制造多样性。
小证据池下多个包可能相同；步骤文件报告不同证据集数和不同药方数，不保证一定产生多样性。

初版没有稠密向量检索、外部搜索、自动扩展知识库或大型图数据库。

## 候选与评分

每个候选输出辨证假设、治法、配伍摘要、逐药理由、原文引用、证据 ID 和不确定性。
全部候选完成后才把原方传给评分器和优化器。模型原始结构化理由保留在步骤文件，事后优化器反思另存。
无效输出保留在分母；网络失败按现有重试设置重跑，仍失败则不提交本步。

评分向量包括：

1. 原方 P/R/F1、最大权一对一匹配的 `soft_f1_optimal`。旧版 greedy soft F1 不改写。
2. 正向药症支持下的症状覆盖、方内有支持药物比例、未知药物及证据 ID。
3. 引用 ID 是否存在于生成时的包内、病例引用是否逐字存在。它们只是引用完整性，不是语义蕴含或解释正确性。
4. 配置中药物配对规则的风险标记、词表外药名数、输出有效性。

评分器从完整固定知识库重新查药症和药物原文，不只采用候选提交的知识。无匹配关系是未知，不判错误。
初版没有自动判断“治法正确”的成熟裁判，也没有把理由长短或术语数量作为奖励。
同病例的 F1 明显改善且风险、引用错误不增加时，记录“原方一致性偏好”。知识支持下降时另存分歧，不把关系缺失当作否定原方证据；优化器仍看到完整评分向量，而不是仅看胜负标签。
这些偏好规则和门控阈值是显式研究代理，不宣称已经校准为临床合理性评分器。

## 排查没有新增技能

```bash
python scripts/evolve_knowledge.py diagnose --run_dir runs/Deprecated/knowledge_test_1
```

只读取已提交的步骤和状态，不调用 API、不加载当前 PTM 数据、不修改旧实验，也不要求旧实验的代码版本与当前相同。
输出逐步候选数、有效候选数、独立描述组数、优化器状态、提出编辑数、门控前拒绝原因和实际门控次数。
v3 起增加 `experience_groups`、`selected_history_groups`、`replayed_cases`，分别表示累计保留的独立经验组、
本步归纳实际选入的历史组、当前批次回放的病例数。`selected_independent_groups` 包含当前批次与历史，不只统计当前 batch。
`effective_gates` 要求至少两个病例的提示词真正改变；旧实验即使执行过比较，`n_affected=0` 也单独计入 `zero_affected_gates`，不能据此说技能无效。
训练返回值也包含 `last_step_diagnostics`。`--max_steps 2` 是累计最多两步，不是两个 epoch；batch=4 时通常只处理 8 例，
即使 `n_practice=16`，此时 `complete=false` 也是正常的。`rules=0` 不等于没有生成候选或没有调用优化器。

优化器字段遵守以下契约，不通过放宽验证来强行保留技能：

- `origin_cases`、`supported_cases`、`counterexamples` 只能填输入中的真实病例 ID；支持与反例不重叠。同一病例的低分候选不是病例级反例。假设的支持列表允许不足或为空，正式升级只用程序独立核验后的支持。
- 低分方案的文字分析、假设性不适用情形放在 `counterexample_notes`，不计入独立病例证据。
- 默认 `semantic`：`kind` 区分程序性原则和病证规则，`applicability` 描述基础范围，`required_conditions` 是必备事实，`decision_steps` 是待执行步骤，`branches` 是可选条件分支，`exclude` 是排除条件。
  `scope` / `retrieval_terms` 只用于召回，不是最终适用判定。
- `principle` 的 `knowledge_ids` 允许为空，但须有真实病例对比，并由定义审核判为纯程序性；涉及医学主张或属于 `pattern` 时仍须有合法的非医案知识引用。
- 定义审核在支持病例检查前执行，防止把辨证步骤或可选表现塞入必要条件。字段混用时反馈错误，不自动删除或移动条件。
- `support_evidence` 为每个声称支持的病例提供逐字原文引用和说明。先核验引用，再独立检查适用性和候选策略差异。未知/不适用的观察原样保留，不计正向支持、不自动扩大适用范围；未来真正满足条件的案例可以补充证据。
- 不同病证分支应拆为不同规则，不把两种病证的条件并成谁也无法满足的列表。独立支持按原始描述组计算，不按 cid 或运行次数计算。
- 格式错误或支持不足的编辑不会进入门控，但保留在假设库；原因、修订标记和已核验支持数回传给下一步优化器。

格式正确仍不保证技能正式生效。过于具体的条件可能无法匹配任何新病例；假设仍保留，只有在新门控病例上表现改善并通过复核才进入正式库。
v3 起在生成门控处方前，先按原文检索有限数量的新描述组，分别确定编辑前后实际注入的技能；v4 同时冻结分支决策。
只有提示词确实改变的组才能入选；门控组、复核组以及两侧适用性结果先保存到 `gate_plans/`，随后才生成处方和评分。
重试同一计划时直接读取冻结结果，不在看到处方或 F1 后重新筛选成员。
若限定预算内没有足够适用组，标记 `insufficient_semantic_coverage_within_budget`，不生成门控处方、不消费组、不记作效果失败。
预筛过程中可能已有适用性模型调用；`examined_groups` 表示实际检查量，不能把有界搜索无命中解读成全训练集无覆盖。
提案持续留在待验证库，不注入普通处方提示词、不算已学会规则；`deferred` 是覆盖不足待验证项的兼容摘要，不是另一套库。
后续步骤自动延续未完成的覆盖扫描；定义不变且真正行为检验失败的方案不自动重试。不会自动把 AND 改成 OR、删除必要条件或降低得分阈值。
修改提示词或实现后须使用新的 run_name，不修改旧记录来补写“已学习”的技能；提示完全相同的候选请求仍可命中既有模型缓存。

### 跨批次经验与技能使用

经验池默认保留最近 256 个独立描述组，包含原文、生成时理由、知识包、原方、多维评分和组内偏好；随步骤原子提交/回滚。
当前有区分度的病例作为锚点，以原文与生成时治法摘要的字符 TF-IDF 检索历史近邻，最多送入 `n_reflect` 组。
相关性只决定经验召回，独立支持数仍另行核验；不把邻近案例当作天然正确的共同规律。
超出优化器输入预算时整例舍弃，不截断原始病例。`literal` 消融保留旧版仅当前批次的归纳方式。

正式技能的使用分两步：字符 TF-IDF 召回至多 `skill_retrieval_k=4` 条，再做逐条语义适用性判断。
程序性 `principle` 可以在低词面相似度时参与召回，但并非自动适用，也受数量上限约束。
判定只接收完整病例原文和技能适用定义，不接收原方、候选处方、奖励或训练支持病例列表。
所有必要事实都要有原文支持；矛盾/明确排除则不适用，必要事实未知则不注入技能。decision_steps 不作为已知事实要求。
可选分支未知不阻止基础技能，但只有具备有效原文依据的分支动作才注入处方提示；其余结构化分支动作不提供给处方模型。
排除项未提到保留 unknown，不冒充已明确排除。引用必须逐字存在；这不等于已经验证语义蕴含。
基础适用性由程序按逐项状态计算，模型总括 decision 另存 reported_decision；不能仅因可选分支未知否决基础技能。
判定保存逐项状态、引用、原因并缓存；练习、门控、冻结评测共用此逻辑。
`knowledge.applicability_max_tokens=0` 沿用模型输出预算，避免推理型模型的短预算空回复。

可用 `--set knowledge.skill_mode=literal` 新建消融运行，保留逐字 AND 条件、OR scope 和 OR 排除逻辑。
该模式保留原拒绝/草稿逻辑，不启用三类库自动补证；数量不足仍记录 `no_fresh_affected_groups` / `insufficient_fresh_affected_groups`，无适用性模型调用。
语义判定本身是可审计的 LLM 判断，不是已经通过临床验证的专家裁判。

## 学习、门控与评测隔离

新的实验协议与旧版全训练集病例 RAG 不同，不能直接横比旧目录的分数。

- 以去标点后的原始描述为组。首先移除与官方 val/test 描述组重叠的训练病例，再划分练习组和门控组。
- `n_practice` 是目标数量，最后一个描述组完整保留，实际数量可能略多；门控每组取一个代表。
- 训练从空医案记忆开始，一步的全部候选及门控完成后才揭晓并加入本步练习案例。自例及同描述医案始终从检索结果排除。
- 每条候选编辑使用新的、提示词实际受影响的门控组和复核组。只有足够完成两阶段验证才分配；分配后即消费，拒绝编辑也不复用。
- 使用 `used_gate_ids` 显式记录已分配组，未匹配的组留给其他编辑，不能用一个位置游标误当已用集合。池耗尽或触发覆盖不足时保留未验证草稿，不强行接受。
- `consumed_gate_groups` 是分配数，不是实际完成评分的病例数；第一阶段失败时，预留复核组仍不复用，但不会调用复核模型。
- 验证规则的原文条件、来源 ID 和独立支持组数；这些检查不声称自动证明模型的医学解释。
- 编辑逐条与当前技能库比较。门控以 F1 改善及引用/风险不恶化为准，复核要求不下降。配对 t 是筛选启发式，不作为多重检验后的显著性声明。
- 每轮采用固定的单个 `eval_view` 验证与评测，默认 integrated。不会按测试原方挑选最佳候选。
- 练习日志中的 `oracle_f1` 仅是事后上界，不能用作推断性能。
- 断点绑定代码、数据、知识库、检索及模型配置。修改这些条件需要新目录，不静默复用旧 run；工作线程数和缓存路径可调整。
- 步骤产物先写、状态最后原子提交。失败步骤回滚已用门控组、草稿、技能、已见医案和经验池。只将 state 中已提交的步骤视为完成。
  已落盘的门控计划保留供同一提案重试，不能视作已通过门控。

目前只报告点估计，没有实现描述组 bootstrap、正式专家标注校准或真实模型效果验证。

## 产物

训练目录保存：`protocol.json`、`config.json`、`knowledge_manifest.json`、`knowledge_snapshot.json`、
`steps/step_*.json`、`state.json`、`hypothesis_library.json`、`pending_library.json`、`skillbook_final.json`、`skillbook_final.md`、模型调用日志。
其中 skillbook_final 只含正式技能。state 的 `skill_libraries` 是三类库的权威断点，保留完整观察、版本、状态转移、归档记录及待验证扫描；导出 JSON 是 train 正常返回时刷新的人读快照。
v3 起另存 `replay_archive.json`、`gate_plans/*.json`；state 内 `experience` 保存跨批次经验，处方输出内 `skill_routing` 保存当时的适用性判定。
v4 步骤另存 `definition_review`；适用性结果含分支状态、`active_branches`、`reported_decision`。定义审核 task 为 `knowledge_skill_review`，会额外调用模型，每个定义缓存一次。
v5 步骤另存 `hypothesis_reviews`、`lifecycle_events`、三类库计数；新增 task=`knowledge_skill_support` 检查候选对比是否支持技能，共用 applicability_max_tokens 预算。正式规则只保存有界的支持摘要，完整证据留在库记录。
步骤文件包含知识包、原始病例、生成时理由、揭晓后的评分、偏好对、编辑与门控逐例输出。
`usage_current_session.json` 是当前调用会话的用量，不是跨所有断点会话的累计费用。
`eval` 自动读取训练配置和已揭晓记忆，允许覆盖 `llm.*` 做显式骨干比较，其余检索条件冻结；评测配置会单独记录。
v5 与 v1/v2/v3/v4 的技能生命周期不同，不修改旧实验的规则数或分数。可以只读诊断或回放旧候选，但不能强行跨协议续跑或把局部门控收益当作完整测试集收益。

```bash
python -m pytest -q tests/test_knowledge.py
python -m pytest -q tests
```
