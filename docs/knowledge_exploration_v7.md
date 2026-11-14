# v7：多方案对比与技能迭代

本版协议为 `knowledge-evolution-v7`。按批次更新的方式不变，修改的是工作技能的使用路径，以及同 F1 方案的知识偏好。模型参数不更新，旧实验目录不改写。

2026-11-12 的真实 pilot 审核修复见第 10 节。修复改变了实现签名，原 `knowledge_v7_pilot64` 目录不能直接续跑，历史候选可用新目录回放。

## 1. 主学习循环

```text
当前批次的病例原始全文 + 当前相关技能 + 不同知识输入
    -> 每个病例生成多个 solution（药方 + 生成时理由）
    -> 全部生成后才揭晓原方，计算 F1 和知识等多维评分
    -> 比较同一病例中的好方案与坏方案
    -> 结合已有技能和相关历史经验，总结、修订、合并或拆分技能
    -> 更新后的工作技能在后续批次继续参与探索候选
```

`batch=8` 仍表示先处理 8 例，再统一反思。当前批次刚看完答案才形成的技能不会回头注入本批次冒充预测；一个病例也不必强制产生新技能。

默认四个候选的组织方式：

| 知识视角 | 可使用的技能 |
| --- | --- |
| relation | 正式库，作为非探索候选 |
| syndrome | 正式库 + 一条适用的工作技能 |
| case | 正式库，作为非探索候选 |
| integrated | 正式库 + 一条适用的工作技能 |

由 `knowledge.exploration_views=[syndrome, integrated]` 配置探索视角。没有适用工作技能的槽位自动使用正式库；有多条适用技能时分配给不同探索槽位，只有一条时可用于不同知识包。每个探索槽位至多使用一条工作技能，避免把大量不成熟规则同时堆入提示。

总候选数仍为 `views × variants_per_view`，默认每例 4 个，不因探索增加。即使把全部视角指定为探索，也保留至少一个非探索候选。不同知识或技能输入不保证处方一定不同，实际差异继续通过日志报告。

## 2. 基本执行条件与成熟度分开

工作技能包括假设库和待验证库中的可执行版本。其进入探索候选只需满足：

1. 有合法的规范化定义，操作为 add 或 replace，未归档。
2. 定义审核通过，包括必要患者事实、步骤、患者分支和过程分支的区分；医学主张仍需合法知识来源。
3. replace 指向的正式目标版本仍然有效。
4. 基于当前完整原文的适用性检查通过；必要患者事实未知时不执行，不由任务场景补造。
5. 被相关性检索召回，且有探索槽位。

不要求已有支持票、达到 `min_support`、配对试用成功或门控通过。单例启发的合法技能可以在下一批次使用。正式晋升失败、容量或正式证据长度问题，也不会单独使一个基本可执行的工作版本失去探索机会；原版本不能重复抽新门控组碰运气的锁仍然保留。

已有 v6 审核修复继续有效：工作场景与患者事实分开，建议与硬错误分开，条件分支不当作无条件命令，明确支持 `revise + revises=H编号`，支持局部而非整套技能的证据描述。

## 3. 为什么以前同 F1 不区分知识好坏

旧 `preference_pairs()` 在 `F1(A)-F1(B) < min_gap` 时直接跳过，因此 F1 相同的方案根本不会进入后面的比较。这是“F1 主监督、知识仅作约束”的保守实现选择，不是 F1 指标本身的限制。

原先的考虑是知识库不完备，缺少药症关系不能当作负面医学证据。但“不能把缺失当错误”不必推出“已有知识支持的差异一律不能参与排序”。v7 增加可审计的同分比较，没有随意拼一个加权总分。

### 当前优劣对规则

- F1 差至少 `min_gap=0.05` 时，沿用原来的 F1 优劣对；风险、引用错误、原文锚点错误、OOV 四项不能变差。
- F1 相同（浮点容差 `1e-12`）时，若双方输出有效，且症状覆盖率、有据用药比例两项均为可比较的已知值，则允许知识比较。
- 两项知识指标均不得下降，且至少一项提升达到 `knowledge_min_gap=0.05`；上述四项错误/风险约束也不得变差。满足时记录 `basis=knowledge_tiebreak` 和每项差值。
- 任一知识值为 `None` 时不当作 0；两项一升一降时不强行判胜负；不允许知识更好悄悄覆盖较低的 F1，也不按解释长度或引用数量加分。
- 知识支持比较由完整固定 KB 计算，不因某个候选恰好拿到更多检索材料就改变评分器的可见知识范围。

这是一条“已有知识支持更充分”的训练偏好，不把未覆盖药物宣布为错误。`knowledge_tiebreak=false` 可关闭新增比较；literal 旧流程不启用它。

## 4. 三类库仍保留

| 库 | 学习与使用 | 验证含义 |
| --- | --- | --- |
| 假设库 | 合法且适用即可参与探索；失败、平分和无效输出都供反思，不只收集成功 | 尚未达到冻结提名条件或需要修订 |
| 待验证库 | 仍可参与探索；可以另建修订版本，但不原地改写当前定义 | 冻结的当前版本等待独立检验 |
| 正式库 | 普通候选及冻结评测使用；后续变更仍另建版本 | 已通过既定独立门控 |

新版本通过 `revises/parents` 保留历史联系，不继承旧版本成绩。工作技能的每组反馈包括其实际候选、有效性、多维分数，以及与其他普通候选的优劣关系。相同原文组只计一次成熟度证据；当前步骤仍保留重复病例的完整输出。

默认 `min_support=3` 只约束“提名待验证”，不约束总结、修订和探索使用。培养模式提名条件是：

- 至少 3 个独立练习原文组有有效探索输出或有效配对对照，二者按原文组合并去重。
- 至少一次普通探索候选的观察性优势，或者一次配对试用的正向 F1 差异。
- 如启用了配对试用且已有有效对照，还要求其平均 F1 差不为负、四项约束总差不恶化。

不同知识包之间的探索差异有混杂，因此这只是“值得独立检验”的提名，不是技能因果效果证明。不会把普通探索记录伪装成同知识包配对试验。

正式晋升仍使用原有独立初筛与复核。主门控依旧以 F1 增益为准，知识同分优势本身不保证成为正式技能；但该工作技能仍可继续参与探索和迭代。冻结评测不使用假设/待验证库。

## 5. 辅助检验与预算

- `hypothesis_trial_budget=0` 是新默认，关闭额外配对试用。设为 2 可额外做每步最多 2 组同知识包试用；它不改变主学习循环。
- `pending_gate_budget=1` 默认每步最多处理一个待验证版本。设为 0 仅推迟正式晋升，不停止工作技能的生成、修订、使用或成熟度记录。
- 工作技能召回共用 `skill_retrieval_k=4` 的候选上限；按原文检查适用性，找到足够填充探索槽位的技能后停止。正式技能路由和定义审核另计调用。
- `collect_only` 关闭工作技能探索、反思、配对试用和门控；`learning_mode=observe` 保留无探索的观察式消融，不是旧协议的精确复现。

## 6. 回放、恢复与审计

- 工作技能选择、知识包、正式基线及探索激活在看到当前分数前写入 `exploration_plans/`。中断后复用冻结计划，不能因先见到某个输出而重新挑选技能。
- 步骤候选增加 `working_skill_id/working_definition_key/working_maturity/candidate_provenance`；`explorations` 连接工作技能与候选索引、评分和偏好。
- 库内 `exploration_observations` 和 `exploration_screenings` 保存训练反馈及未知条件，供后续修订；失败和平分也会进入优化器材料选择。
- 新增诊断 `exploration_cases/exploration_candidates/knowledge_preference_pairs`。`rules` 仍只统计正式技能，不能单独代表训练是否在学习。
- `--reuse_run` 允许导入 v1-v7 的兼容历史候选。历史探索输出只作为有来源的旧方案，不继承旧技能、旧成熟度或配对信用。
- 对已有回放病例，非探索槽位可以继续复用；一旦选中当前工作技能，对应探索槽位必须按当前策略重新生成，并标记 `mixed_replay_exploration`。不能给旧处方贴上新 H 编号冒充新技能输出。
- `--replay_only` 要求病例有历史候选，不意味着完全不调用模型；探索槽位、反思、适用性和启用时的验证仍会产生请求。
- 旧目录不直接续跑新协议，必须使用新 `run_name`。批次内生成后再评分、练习与门控分区、val/test 隔离、传输失败回滚和完整原文输入均保留。

## 7. 小规模训练命令

在项目根目录运行，复用已有 v5 的 32 例候选：

```bash
export DEEPSEEK_API_KEY=magpie
python scripts/evolve_knowledge.py train \
  --llm openai_compatible \
  --run_name knowledge_v7_pilot64 \
  --reuse_run "$PWD/runs/knowledge_v5_pilot64" \
  --set knowledge.n_practice=64 --set knowledge.batch=8 \
  --set 'knowledge.exploration_views=[syndrome, integrated]' \
  --set knowledge.hypothesis_trial_budget=0 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 \
  --set knowledge.gate_candidate_limit=24 \
  --set knowledge.min_support=3 --set knowledge.pending_gate_budget=1 \
  --set knowledge.knowledge_tiebreak=true \
  --max_steps 4
```

仅当源码与实验配置均未改变时，同配置改成 `--max_steps 8` 才能从对应断点继续到总计 8 步。下述审核修复后的运行请使用第 10 节的新目录命令。若只想先检验主学习循环，可在新实验启动时设置 `pending_gate_budget=0`，不会阻止探索或修订；中途改配置需另建运行目录。

```bash
python scripts/evolve_knowledge.py diagnose --run_dir runs/knowledge_v7_pilot64
```

## 8. 实现边界

主要新增 `knowledge_exploration.py`；`run_step()` 保持批次顺序，`preference_pairs()` 增加同 F1 知识比较，`SkillLibraries` 区分探索观察与可选配对证据。

基本适用性及定义语义仍由 LLM 判断，精确引用和结构检查不能消除全部误判；学习过程不保证单调改善。流程测试、模拟后端结果和训练期候选上界都不能代替真实模型的冻结评测。

## 9. 本次验证（2026-11-12）

- 全量测试 `302 passed`，新增 28 项探索/知识偏好回归测试；修改模块 Ruff 和 `git diff --check` 通过。
- 实际 PTM 数据上的 mock 运行：`runs/knowledge_v7_exploration_mock_20261112`，先跑 2 步，再原配置续到 4 步；`batch=8`，配对预算和门控预算均为 0。
- 共处理 32 例、128 个候选：80 个历史候选直接复用，48 个探索候选重新生成，覆盖后续 24 例。每批仍是 32 个候选，没有额外配对输出。
- 产生 39 对同 F1 的知识偏好。第 2 步形成的 1 个待验证技能在第 3、4 步继续用于探索，正式技能仍为 0；这验证学习使用不依赖正式晋升，而不是证明真实技能收益。
- `runs/knowledge_v7_exploration_mock_eval_20261112` 完成 3 例冻结评测，均未注入工作技能；`diagnose` 读取成功。
- v5 源运行的 state、protocol、4 个步骤文件及原 v6 mock state 的哈希未改变。未调用真实 LLM API。

## 10. 真实 pilot 的审核修复（2026-11-12）

`knowledge_v7_pilot64` 的 H1/H2/H4/H5 曾因审核器混淆分支条件、动作和医学证据类型而被拦截；H3 在 `tr26485` 上因未给具体年龄、未明确否定排除项而被判未知。修复不删除必要事实、不降低 `min_support`，也不改变批次更新、三库成熟度或独立门控。

### 分支条件与证据类型分开

- 主审核明确 `branch_roles` 只分类 `when`。患者条件后接选药动作仍是 patient；候选药功效比较仍是 procedure，医学主张的来源检查另行执行。
- 主审核的分支分类与声明不一致且没有其他明确 blocking issue 时，执行一次 `knowledge_branch_conditions` 独立复核。输入只有有争议分支的 index/when，不提供 actions、技能全文、作者声明 kind 或 evidence_type，避免这几项相互干扰。
- 输出的患者条件片段和方案条件片段必须逐字来自 when。真正的混合条件须给两类不互相包含的片段；分类仍不匹配、仍不确定或响应无效时，技能仍不能执行。
- 原分类保留在 `reported_branch_roles`，复核输入、引用、请求标识、结果和校验错误保留在 `branch_condition_recheck`。不会把 medical/mixed 主张自动变成 procedural，也不会免除知识来源要求。

### 基础适用范围逐段复核

- 正向必要事实、全覆盖人群措辞、排除情形分别解释。“成人与小儿”不要求具体年龄；“仅小儿”“孕期”等限制仍需病例证据。
- 只有基础范围 `requirements[0]` 为 unknown、其余 required_conditions 全部 supported、没有明确成立的排除项且原引用校验通过时，才执行一次 `knowledge_scope_review`。其他必要事实缺失、明确矛盾或排除项成立不能借此绕过。
- 复核将原 applicability 拆为 patient_condition / unrestricted_scope / exclusion / unclear。各片段按顺序拼接必须与原文完全一致，不能漏掉必要条件；患者事实和排除判断必须引用病例原文，全覆盖声明只能引用技能范围自身，不能冒充患者事实。
- patient_condition 的未知仍阻断；明确排除情形成立仍阻断；未知 exclusion 不要求额外证明不存在。真正“已排除某风险才可使用”的必要否定仍作为 patient_condition，而非普通排除项。
- 原判断保留在 `reported_requirements`，分段依据保留在 `scope_recheck`。不更改 required_conditions、已知排除项或患者分支激活状态。
- 两种复核均有缓存，每次审核最多追加一次对应请求；传输失败中止当前步骤而不是记成负例。结构或引用无效不放行。不保证 LLM 从此没有语义误判。

### 修复验证

- 新增 39 项定向回归，全量测试 `341 passed in 86.18s`；Ruff 和 `git diff --check` 通过。
- 使用当前配置的真实 API，对历史首轮错误判断触发新复核，共 6 次请求。H1/H2/H4/H5 的分支分类误判均得到纠正；H3 在 `tr26485` 上由 uncertain 变为 applicable，原排除项的 unknown 状态未被改成 absent。
- 第一次范围复核仍错误声称没有病例，并把单独分号当未知项。补充完整原文的输入含义、已有必要条件引用及纯标点处理后再测通过。初次失败与最终结果均保留在 `runs/knowledge_v7_audit_fix_verification_20261112/`，最终汇总为 `final_report.json`。这只是历史错误的定向回归，不是完整重新训练或性能提升证据。
- 当前源码下的真实 CLI mock 冒烟：`runs/knowledge_v7_auditfix_mock_20261112`，先运行 1 步，再恢复到 2 步，共 16 例。第 2 批有 16 个探索候选、1 个待验证技能；仅证明流程和恢复可运行，不代表真实模型收益。
- 原 `knowledge_v7_pilot64` 的 state、hypothesis_library、protocol 文件哈希未改变；未运行大规模真实训练或 500 例测试。

### 修复后重跑

不要删除旧目录或手改旧 state。用新运行目录复用 v7 已有的 32 例候选：

```bash
export DEEPSEEK_API_KEY=magpie
python scripts/evolve_knowledge.py train \
  --llm openai_compatible \
  --run_name knowledge_v7_auditfix64 \
  --reuse_run "$PWD/runs/knowledge_v7_pilot64" \
  --set knowledge.n_practice=64 --set knowledge.batch=8 \
  --set 'knowledge.exploration_views=[syndrome, integrated]' \
  --set knowledge.hypothesis_trial_budget=0 \
  --set knowledge.gate_size=4 --set knowledge.confirm_size=4 \
  --set knowledge.gate_candidate_limit=24 \
  --set knowledge.min_support=3 --set knowledge.pending_gate_budget=1 \
  --set knowledge.knowledge_tiebreak=true \
  --max_steps 8
```

这是新实验，不是迁移旧断点：复用历史候选及暴露记录，但重新归纳、审核和培养技能，不继承旧技能的审核结论或成熟度成绩。预计处理 64 例练习病例（分组抽样可能略超目标）；门控病例另计。没有在此命令中运行 500 例测试。

## 11. 训练进度显示

训练 CLI 默认向 stderr 显示进度，不需要修改已有训练命令：

- 总进度为已完成批次的病例数及当前批次位置，断点恢复从已有 cursor 开始；失败批次不会提前计入完成数。
- 生成方案时显示当前生成任务的完成数/总数；回放的历史方案不会冒充新请求，分组生成任务分别计数。
- 显示知识匹配、方案生成、技能总结、定义审核和验证等阶段。模型请求等待期间持续显示已等待秒数，不虚构请求完成百分比或 ETA。
- 正常终端原行刷新；日志重定向时使用普通文本，阶段切换立即输出，等待期间约每 10 秒输出一次。最终 JSON 仍单独写 stdout。
- `--no_progress` 只关闭显示，不改变实验配置或签名；达到 `--max_steps` 但尚未训练完时，不会显示总进度 100%。

进度模块 `knowledge_progress.py` 只在 CLI 端代理请求和观察批次，不修改算法模块、提示、随机数、门控或已保存结果。此次显示修改不要求换 run_name，也不会使此前审核修复后的断点失效；此前审核修复本身的版本限制仍然有效。

实测使用 mock 后端在 `runs/knowledge_v7_progress_smoke_20261112` 先运行 8 例，再恢复到 16 例，终端可见生成任务进度与完成状态；与相同配置的修复后旧 mock 运行相比，protocol_id 和完整 state.json 内容均一致。新增测试覆盖串行/并行请求、等待刷新、失败清理、断点恢复、关闭显示和 stdout JSON 隔离；未为进度测试调用真实 API。

## 12. 冻结工作技能的配对测试

新增独立入口 `scripts/eval_working_skills.py`。它不重新训练、不晋升技能、不修改原运行，且不改变旧 `evolve_knowledge.py eval` 只测正式库的含义。新测试不要求正式技能数量大于零。

### 比较的两组

- `baseline`：原运行的冻结正式技能库。当前 `knowledge_v7_auditfix_n500_seed42` 正式库为空，因此是无技能组，但仍有知识检索和已见训练病例记忆。
- `working`：同一正式库，按病例原文检索工作技能，按相关性顺序检查适用性，使用第一个适用的版本，最多一个。`replace` 保留原本替换正式技能的语义。每例不生成多个药方再取最高分。
- 工作版本复用训练时 `working_entries(include_pending=True)` 的筛选：可执行、定义审核通过、未归档、替换目标未失效、应用版本关系；不增加支持数、配对试用或门控通过要求。可执行但门控失败的版本仍按训练策略保留，并单独统计。
- 检索数量、语义匹配配置、知识视角、模型参数均读取保存配置。当前配置最多检索 4 条，知识视角为 `integrated`。
- 两组使用相同完整病例原文、同一知识包和同一冻结训练记忆，各输出一个药方及理由。没有适用工作技能、两组提示完全相同时，直接复用基线输出，避免把重复采样波动记作技能收益。
- 选择和适用性判断不使用测试原方。两组输出都保存后才按原方及知识评分，评分不会回写技能或训练记忆。工作组增加适用性判断调用，因此不是总 API 预算相同的比较。

### 运行

直接对同一固定 `test_order` 的前 500 例配对评测，不要重新执行包含训练的 `train_v1.sh`：

```bash
export DEEPSEEK_API_KEY=magpie
DIR="$PWD/runs/knowledge_v7_auditfix_n500_seed42"
python scripts/eval_working_skills.py \
  --run_dir "$DIR" \
  --llm openai_compatible \
  --split test --n 500 \
  --out "$DIR/test500_working_paired_v1"
```

追加 `--prepare_only` 可以只校验训练协议、数据、源码和库，冻结清单，不调用 LLM。去掉它并重复同一命令即可开始。需要 mock 流程检查时使用 `--llm mock --n 4` 和另一个输出目录，不把 mock 数值当模型性能。

默认有评测进度，`--no_progress` 关闭。每例检索/适用性计划先冻结，基线和工作组分别落盘；中断后同命令恢复，已有计划和输出不重新请求。请求重试耗尽时中断，不以 F1=0 代替网络失败；模型成功响应但处方无效仍按原评分逻辑记为失败。相同输出目录不允许混用另一 checkpoint、病例清单、参数或评测代码，不允许并发写入。

### 查看结果

`metrics.json` 中 `complete=true` 才代表全部跑完；中途显示的是已完成前缀：

- `all.baseline.f1`、`all.working.f1`、`all.delta_f1`：全部 500 例的配对比较。
- `activation_n`、`activation_rate`：实际注入工作技能、提示发生变化的病例数和比例，不是模型自报已遵守技能的比例。
- `activated.baseline.f1`、`activated.working.f1`、`activated.delta_f1`：同一触发子集两组比较，不能拿它与全体基线直接比较。
- `gate_failed_activated`：曾门控失败但仍可执行的版本实际触发病例上的配对指标。
- 同时输出 precision/recall、soft F1、知识支持、引用/原文锚点、有效输出等指标；`known_n` 记录各维度有值的分母，未知不填零。

`protocol.json` 记录固定病例 ID/原文签名、来源 checkpoint、模型、源码和评测策略；`snapshot.json` 保存三库和训练记忆 ID；`cases/` 保存逐例计划、两组输出与评分；`predictions.json` 为完成后的汇总，`usage_sessions/` 保存各次运行的请求消耗。

这项结果应标为“冻结工作技能库的探索性测试”，不是“已验证正式技能的效果”。不能看完测试分数再挑技能、改路由参数或改版本，然后继续把同一测试集作为未使用的最终验证。

### 本次验证（2026-11-13）

- 对真实 500 例训练 checkpoint 只做离线校验：189 条库记录，100 条可执行工作版本，其中 2 条曾门控失败，正式技能 0 条。计划中的 500 个测试 ID 及顺序与 `runs/Deprecated/zeroshot_cot_n2000_random_deepseek/deepseek-v4.1-flash/test_preds.jsonl` 完全一致。
- 真实 CLI 使用 mock 跑 4 例，验证双组生成、进度、评分及恢复；再次执行同命令新增请求为 0。产物位于 `runs/knowledge_working_eval_verify_20261113/`，mock 分数不代表真实模型性能。
- 全量测试 `372 passed in 106.51s`，包含 22 项新增工作技能评测测试；Ruff 与 `git diff --check` 通过。原运行的 state、protocol、config 文件 SHA256 不变。
- 未调用真实 API 执行 500 例测试；上面的命令由使用者启动真实评测。
