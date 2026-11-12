# 知识技能进化：判断条件与实现清单

> 历史清单：下面保留 v5 的判断语义。当前的技能使用、学习与成熟度条件见 [v7 多方案对比与技能迭代](knowledge_exploration_v7.md)；此前审核修复见 [v6 说明](knowledge_cultivation_v6.md)。不要直接用本页的升级条件解释新版本日志。

本清单对应 `knowledge-evolution-v5` 的 semantic 模式。区分程序硬校验、LLM 语义判断、数值筛选和仅记录的诊断；没有把这些检查描述为独立临床验证。
默认值来自 `configs/default.yaml`；命令行 `--set` 会覆盖。`knowledge_test_4` 实际使用了 min_support=1、gate_size=4、confirm_size=4，不是默认值。

## 0. 三类库与升级条件

实现：`SkillLibraries`、`ContrastSupportJudge`、`KnowledgeEvolver._remember_hypothesis()` / `_advance_libraries()`。

| 当前情况 | 归属/处理 | 是否参与普通开方 |
| --- | --- | --- |
| 想法来自现有病例，但必要事实未知、支持不足 | 假设库，等待后续练习补证；unknown 不是反证 | 否 |
| 格式错误、引用非法、医学性主张缺 KB 来源、必要条件与步骤混用 | 假设库，needs_revision=true，保留原稿与具体原因供优化器修订 | 否 |
| 定义合法，达到 min_support 个独立组的“适用 + 对比支持” | 自动转待验证库 | 否 |
| 待验证方案暂时找不到足够新门控组 | 留待验证库，后续步骤继续扫描未检查的描述组 | 否 |
| 独立初筛和启用时的复核通过 | 正式技能库；每例仍要检查适用性 | 是 |
| 真正行为检验失败 | 回假设库，gate_failed/needs_revision=true，不自动重试不变的定义 | 否 |
| 旧假设被新定义修订，或正式技能被门控通过的替换/删除移除 | 原记录归档留在假设库，不擦除历史 | 否 |

每步先生成并评分当前练习，再接收新提案、审核定义、补充假设证据、处理待验证队列，最后一起提交断点。即使本步优化器没有新提案，历史库也会按预算推进。
假设的独立支持检查从当前练习和已提交经验池选取；相关原文 TF-IDF、提案来源/声称支持病例优先，程序性原则可在低词面相关时候选；不读取未来练习/门控/val/test 充当支持。
每组先检查必要事实，再以已记录的组内优劣对、生成时理由、知识包、评分判断是否 `supports/opposes/unclear`。合法对比索引和逐字原文引用由程序核验；解释是否成立仍由 LLM 判断，不等于证明因果。
仅 `applicable + supports` 计正向支持。同一规范化原文描述组固定保留一次观察，重复提案/同描述 cid 不多计；反向和未知观察同时保留，但本版不额外设置反向证据比例的硬否决阈值。

默认每步 `hypothesis_review_budget=12` 个假设-病例检查组合，每个组合至多两次 LLM 请求；按最久未检查顺序轮转，达到支持数就停止继续攒正例。待验证门控/预筛预算 `pending_gate_budget=1`，按最久未尝试顺序处理。
定义审核额外计费，按不同模型输入缓存；不计入上述 12 对预算。原文、支持观察、门控扫描均持久化，断点恢复校验练习来源、定义一致性、已验证支持数及正式库对应关系。

提案去重键包含规范化后的操作、目标版本、文本、适用边界和步骤等核心定义，不含支持列表/引用等证据元数据；它是确定性去重，不是语义等价检测。
`revises: "H编号"` 关联修订，核心定义改变产生新条目、重新核验支持；旧条目归档。换支持列表不能清除失败锁。replace/delete 绑定目标正式规则版本，目标已变则回假设库要求重新提案。
删除操作不新增技能，沿用无需支持病例/定义审核的规则，进入待验证队列后照常门控。

权威断点为 `state.skill_libraries`；`hypothesis_library.json` 和 `pending_library.json` 是正常结束训练调用时的导出，正式库仍为 `skillbook_final.json/.md`。`complete` 只表示练习流结束，库内可以仍有未验证项。

## 1. 继承的两个契约

### 1.1 程序性技能允许以病例对比为证据

实现：`KnowledgeEvolver.validate_edit()`、`check_definition()`；结果保存在 `details.evidence_basis` 和步骤的 `definition_review`。

| 情形 | 程序判断 | 后续检查 |
| --- | --- | --- |
| principle，knowledge_ids 为空 | 可以先作为假设；升级时支持仍须满足 min_support | 定义审核必须判为 procedural，适用性、对比支持与行为门控照常执行 |
| principle，knowledge_ids 非空 | 所有 ID 必须来自本次优化器输入的非 case 知识证据 | 程序性或医学性内容均可进入后续检查；不等于引用已证明该主张 |
| pattern，knowledge_ids 为空 | 保留假设；定义审核后阻止升级，标记 medical_claims_require_knowledge | 等待补充合法来源，不进入支持检查或门控 |
| 任意类型，含未知 ID、错误类型 ID、case 医案 ID | 留假设库待修订：invalid_knowledge_evidence | 不因其为 principle 而忽略错误引用 |
| 声明 principle，但定义审核认为含医学主张且无知识 ID | 留假设库待修订：medical_claims_require_knowledge | 不允许仅改类型名绕过知识要求 |

`case_contrast` 表示病例对比来源；`case_contrast_and_knowledge` 表示同时列有知识来源。程序性策略可以写主次取舍、证据核对、避免无据扩展；具体药效、替代、配伍、禁忌、确定性病机或治疗主张不自动归入纯程序性。
类型划分由一次独立 LLM 定义审核判断，可能误判；程序只保证执行该结果和记录依据。没有用关键词表假装完成医学语义分类。
支持病例有偏好对不等于该技能已解释差异的原因；v5 增加对比支持审核检查观察性对应，该因果关系仍没有自动证明。

### 1.2 必备事实、操作步骤、分支与排除项分开

可升级提案必须使用如下字段；旧 `conditions` / `actions` 顶层字段标记待修订，不能悄悄迁移或放宽其含义。

| 字段 | 含义 | 是否参与基础适用性门槛 |
| --- | --- | --- |
| applicability | 基础适用范围，不含可选分支的限定 | 是，作为第 0 个必要项 |
| required_conditions | 进入技能前必须由病例支持的事实 | 是，每项都须 supported |
| decision_steps | 后续要执行的辨证、比较、核对、选药步骤 | 否，不要求病例预先给出执行结果 |
| branches[].when | 某可选动作的触发条件 | 否，只决定该分支是否可注入 |
| branches[].actions | 对应分支的动作列表 | 只有 when=supported 且原文引用有效时注入 |
| exclude | 明确出现时阻止基础技能的情形 | present 阻止；unknown 不视作已经排除，也不自动阻止 |

例如“进一步辨寒热虚实”应为 decision_steps，不应要求原文先提供完整辨证结论；“若出现某兼症，再核对相关证据”应是 branches。
如果某操作确实要求先明确排除风险，仍应把“已明确排除”放进 required_conditions。不能为了扩大覆盖把必要条件移成可选项。

定义审核逐项判断适用范围和 required_conditions 的角色，发现 decision_step、optional_branch、错误放置的 exclusion 或 unclear 时阻止升级、保留为待修订假设，反馈具体字段和原因给下轮优化器。也检查把分支动作重新写成无条件命令的情况。
审核不自动删除、移动或改写条件。模型传输失败抛异常，定义审核 JSON 不合法则记录 invalid_definition_review，不把它说成技能已被效果实验否定。

实现：`SkillRouter.review_definition()`；每个不同定义缓存一次，task=`knowledge_skill_review`，共用 `applicability_max_tokens` 输出预算。
审核输入没有目标病例、原方、候选得分或支持病例列表；它检查定义角色，不负责证明医学事实或引用蕴含。

## 2. 数据与分区判断

实现：`raw_text()`、`grouped_partition()`、`read_replay_sources()`、`replay_compatible()`、`restore()`。

| 判断条件 | 实现和处理 |
| --- | --- |
| 原始描述存在 | desc_raw 必须是非空字符串，否则报错；不回退到抽取症状或清洗描述，原文不截断 |
| 描述组隔离 | 使用 text_key 去掉标点/空白后分组；不是语义近重复判定，不保证不同表述的同一患者被识别 |
| 训练与评测隔离 | 与官方 val/test 描述组重叠的训练病例不进入练习或门控池 |
| 练习/门控分开 | 按 seed 打乱描述组，练习目标默认 480 例，最后一组完整保留；剩余每组一个代表进入门控池 |
| 医案记忆只含已揭晓案例 | 当前 batch 结束并提交后才加入；生成时排除自例及同描述案例；门控病例不加入医案或经验池 |
| 历史回放兼容 | 校验来源协议、数据签名、KB 版本、原文、原方、已提交步骤次序；只导入练习候选，不导入旧技能或门控处方 |
| 回放不借用未来信息 | 候选引用的医案必须在新流中已揭晓，且不同于当前描述组；不兼容时重新生成，replay_only 则报错 |
| 避免重复试验 | 来源已见练习组、已分配门控组以及上游回放曝光记录从新门控池排除 |
| 回放不挑赢家 | 同 cid 多来源固定取首个来源，不按原方得分择优；保留历史策略来源标签 |
| 断点有效性 | seen_ids 必须是已提交练习前缀；used_gate_ids 不重复且属于门控池；经验只能来自已见病例，原文、原方和分组均核验 |

## 3. 检索、候选与输出有效性

实现：`KnowledgeRetriever`、`KnowledgePolicy._answer()`、继承的 `SkillPolicy.run()`。

- 默认 relation / syndrome / case / integrated 四种知识视角，每种 1 个候选。变体轮换相关证据，不通过看原方挑知识或提高温度冒充结构多样性。
- 原文字符 TF-IDF、药症关系和已揭晓医案三路召回；只有限扩展“症状→药物→原文”。否定识别是词表/前缀启发式，不是完整语义解析。
- 每包默认最多 12 项、9000 个序列化字符；摘录默认 450 字符，每药最多 2 项；原始病例不计入此截断预算。
- 生成完成前不向模型提供当前原方。之后评分器和优化器可见原方。
- 处方必须是 JSON 对象；herbs 非空且各项为对象；syndrome_hypothesis、principle、formula_reason、逐药 reason 为非空字符串。
- 每味药要有字符串 name，以及列表 case_quotes、evidence_ids；全局 case_evidence 必须非空列表。
- 药名经既有词表和别名表规范化、去重；至少 1 味有效药，最多 prescribe.max_herbs=30 味。词表外名称记录为 OOV，不因为有 1 个 OOV 就把整个有效处方删掉。
- 引用 ID 必须在该候选生成时的知识包里；非空原文引用必须逐字存在。错误记入 citation_errors / anchor_errors，不自动把整个处方置为无效。
- 当前逐药引用列表可以为空；没有知识引用时引用有效率为 None，而非 0 或 1。这是已实现边界，不能说所有逐药理由均已验证。
- 网络/HTTP 失败按客户端和处方层重试配置处理，最终失败中断并回滚本步，不作为 F1=0 的普通样本。响应成功但处方格式无效则保留无效样本，F1 计 0。

## 4. 评分与“好/坏”偏好

实现：`KnowledgeJudge.evaluate()`、`optimal_soft_f1()`、`preference_pairs()`。

| 量 | 算法 | 用途 |
| --- | --- | --- |
| 原方 P/R/F1 | 规范化药物集合交集；无效输出 F1=0 | 组内偏好与行为门控的主指标 |
| soft_f1_optimal | 精确命中 1 分；剩余同功效类别药物以 0.5 分做最大权一对一匹配 | 记录及优化器参考，不是门控主指标 |
| symptom_coverage | 全文中被识别为非否定的去重症状词，有多少得到方内药物关系覆盖 | 记录；没有可识别正向词则 None |
| supported_herb_fraction | 方内有至少一条当前正向药症关系的药物数/方内药物数 | 记录；关系缺失表示未知，不是错误 |
| citation / anchor validity | 引用 ID 存在率、原文逐字引用有效率 | 记录并以错误次数参与约束；不检查医学蕴含 |
| risk_flags | 配置配伍规则命中的去重药物对数 | 相对风险约束；没有剂量、炮制、患者特异禁忌的完整安全评估 |
| oracle_f1 | 同病例候选中事后最大 F1 | 仅诊断上界，不可当部署性能或选方策略 |

组内偏好 A > B 需要同时满足：

1. A 为有效输出；A 和 B 是同一病例的不同候选。B 可以是格式无效的候选。
2. F1(A)-F1(B) >= knowledge.min_gap，默认 0.05，等于阈值也保留。
3. A 的 risk_flags、citation_errors、anchor_errors、oov_count 四项均不高于 B。
4. 症状覆盖或有据用药比例下降超过 0.05 时，只记录 knowledge_support_disagreements，不自动否决偏好，因为关系缺失不构成反证。

没有按跨病例绝对 F1 把“易题”当成“好策略”。目前也没有合成临床总分、黄金处方多参考最高分、核心药加权、剂量级评分或独立专家标签。

## 5. 哪些经验进入归纳

实现：`ExperiencePool.add()/select()`、`KnowledgeEvolver._optimizer_context()`。

- 经验默认保留最近 256 个不同描述组，首次保存的同组代表不会因重复运行而增加独立支持数。
- 只有存在组内偏好对的病例参与归纳；无差异病例保留日志，但不据此归纳胜负经验。
- 当前有区分度的病例作为锚点，用原文和候选生成时的治法摘要检索历史相关病例，默认最多 n_reflect=6 组。
- 输入连同书库、拒绝反馈、知识包超过 optimizer_max_chars=120000 字符时舍弃整例，不截断病例原文；因此实际送入可以少于 6。
- v5 选入至少 1 个独立有区分度组即可调用优化器提出假设，不再将 min_support 当作产生想法的门槛。默认 min_support=3 仍约束升级；显式设为 1 时确实放宽至单例，本次没有另加隐藏的“至少 3”门槛。
- 优化器每次最多 k_edit=3 条编辑，多出的不处理。允许输出空编辑，不为得到 rules>0 而强制生成。

## 6. 提案结构与书库校验

实现：`KnowledgeEvolver.validate_edit()`、`SkillBook.apply()`。

1. 只接受 add/replace/delete。delete 仍须指向存在的规则，并通过行为门控；它不需要新增支持病例或定义审核。
2. add/replace 的 origin_cases/supported_cases/counterexamples 必须是本次优化器实际看到的 cid，列表内容有效，支持与反例不重叠。库会另存本次输入来源，不把“产生灵感”计为支持。
3. 假设的 supported_cases 可为空或不足 min_support。升级时由程序根据实际检查记录重建支持，按描述组去重后达到 min_support；同一描述的多个 cid 不多计。
4. support_evidence 对每个声称支持的 cid 恰有一项，包含非空 reason 和至少一个逐字存在于其完整原文中的 quote；列表为空时不得编造引用。旧声称与新核验观察分别保留。
5. 没有真实反例允许 counterexamples=[]。同病例低分候选的说明放 counterexample_notes，不冒充病例级反例。当前只验证反例 ID，不自动证明该病例确实反驳技能。
6. 知识依据遵守第 1.1 节；实际正向支持必须有非空 preferences，且对比审核引用合法偏好对。未证明该技能就是候选差异的因果解释。
7. kind、applicability、retrieval_terms、required_conditions、decision_steps、exclude、branches 必须符合新结构；decision_steps 至少一项，分支 when 非空且 actions 至少一项。必要条件、分支、排除项可为空列表。
8. principle 非空；pathogenesis_hypothesis/principle/uncertainty 分别最多保存 600 字符；结构详情序列化超过 5000 字符则保留原稿、标记待修订，不进入待验证。正式升级再次检查大小；只存 min_support 个有界支持摘要，完整证据留库内。
9. SkillBook 将主规则 text 清理并截至 200 字符；不足 6 字符则留假设库待修订。这个入口的 200 是显式代码值，不是旧 skillbook.max_rule_chars=100。
10. scope 只保留长度 1–8 的词，最多 4 个；semantic 实际召回还使用独立 retrieval_terms，scope 不作逐字入场门槛。
11. 应用至正式书库时，主文本去空白后的字符 bigram Jaccard >=0.7 视为重复，不加入并回假设库待修订；这不是医学语义等价判定。假设库自身用第 0 节的完整定义键去重。
12. 无 scope 的规则默认上限 20，有 scope 的规则默认上限 80；replace 必须存在、不得与其他规则重复。新增受容量检查，现有 replace 路径不额外检查跨分类后的容量，此轮未调整。

每个已处理提案均先落假设库。可执行定义依次经过：定义角色审核→已观察练习的适用性与对比支持→足够独立支持后入待验证→新病例门控。前一阶段不足不消耗后续阶段调用，但记录仍保留。

## 7. 语义适用性和分支判断

实现：`SkillRouter.judge()/resolve()`、`KnowledgePolicy.render_rule()`。

| 检查结果 | 程序执行 |
| --- | --- |
| 任何必要项 contradicted，或任意排除项 present | not_applicable，不注入技能 |
| 必要项有 unknown | uncertain，不注入技能 |
| 必要项全部 supported，排除项无 present | applicable，即使某些排除项 unknown 或可选分支 unknown |
| 必要/排除列表不完整、下标不合法、状态非法、理由缺失或非 unknown 项引用无效 | uncertain，不注入技能 |
| 分支 supported 且引用有效 | 仅在基础技能 applicable 时将该分支写入 active_branches |
| 分支 unknown/contradicted | 不注入该分支，不阻止基础技能 |
| 分支输出存在格式/引用错误 | 本次所有可选分支停用，单独记录 branch_validation_errors，不由此否决基础技能 |

基础 decision/reason 由程序根据逐项状态计算；模型的总括结论另存 reported_decision/reported_reason，并以 decision_overridden 标记两者是否不同，不能在各必要项有据时仅因可选项缺失把技能判成不适用。
模型仍负责逐项语义状态，因此逐字 quote 检查只能证明引文存在，不能保证它真正支持该状态。
未知排除项会作为 unresolved_exclusions 告知处方模型，不能被说成已排除。

正式使用前先用字符 TF-IDF 召回至多 skill_retrieval_k=4 条；最低分默认 0.02，principle 可越过该相似度下限参与排序，但不保证一定进入前 4，也不自动适用。
判断只看完整病例原文和技能定义，不看原方、候选处方、得分或训练支持病例列表。练习、支持检查、门控、冻结评测共用判定逻辑；支持检查直接检验被引用规则，不经 top-k 筛除。
每个计数的正向支持病例都必须 applicable 且对比结果 supports。某个灵感来源 uncertain 不再否定整个假设，它保留为未知观察；只有其他真正支持的独立病例足够时才能升级，不删必要条件，不把未知观察伪装成已支持。

## 8. 门控与接受条件

实现：`_semantic_gate_plan()`、`frozen_gate_plan()`、`_compare()`、`gate()`、`paired_stats()`。

1. 只在从未分配过的门控描述组中，按更改技能与完整原文检索；每次预筛默认最多新增检查 gate_candidate_limit=128 组，不按原方或处方得分挑样本。相同上下文延续上次未完成扫描，复用尚未消费的适用组；正式库或操作定义变化会重置扫描。
2. 先分别求编辑前后实际注入的技能/分支，只有最终提示词变化的组才 eligible；模型自报 used_rules 不是入选依据。
3. 必须凑齐 gate_size+confirm_size 个不同新组才分配。默认 48+48；knowledge_test_4 是 4+4。不足则 deferred，不评分、不计为效果失败。
4. 成员和两侧适用性/分支决策先落盘，再生成处方。相同计划重试读取冻结文件，不看结果后改成员。
5. 初筛：n_affected>=2，平均 ΔF1>=0.005，配对 t>=1.28，四项 guard 总次数均不增加。
6. t 使用逐病例 F1 差值均值除以样本标准差/√n；差值方差为 0 且均值>0 时原值为 +inf，日志保存 99，否则无正增益时为 0。
7. 初筛通过才做复核：n_affected>=2，平均 ΔF1>=0，四项 guard 总次数均不增加。复核不另要求 t>=1.28。
8. guard 是 risk_flags、citation_errors、anchor_errors、oov_count；当前按本阶段病例的总次数比较，不保证每个病例逐项不增加，也不是“出现任意风险即淘汰”。此轮没有悄悄改变这个规则。
9. 初筛失败时不调用复核处方，但预留的复核组仍记为已分配，不能再试另一技能。consumed_gate_groups 是分配量，不是模型调用量。
10. 通过两阶段才加入正式技能库；只通过定义/适用性不算学会。候选知识覆盖、soft F1 目前不直接决定接受。

配置允许 confirm_size=0 关闭复核，此时初筛通过即可接受；启用的初筛、复核各至少 2 组。默认没有关闭复核。

配对 t 和门控是研究筛选启发式，不是多次尝试校正后的统计显著性结论。4+4 小试无法证明稳定收益。
deferred 项不会自动生效；v5 自动在后续步骤推进待验证队列，不需要重复提案。有界搜索无覆盖不能声称整个训练集都无覆盖。真正行为门控失败则回假设库待修订，不再让同一定义自动抽新组重试。

## 9. 失败、评测与审计

- 定义审核使用 task=knowledge_skill_review；适用性使用 knowledge_applicability；v5 对比支持审核使用 knowledge_skill_support。即使 consumed_gate_groups=0，仍可能已有这些调用。
- 三种审核共用 applicability_max_tokens，默认 0 表示继承模型预算；不因预算过小的无效 JSON 静默接受技能。
- 定义、适用性和对比审核传输失败中断本步；无效响应按对应校验原因待修订/置为未知或 unclear，不冒充零分处方。
- 步骤文件先保存、state 最后原子提交；失败一起回滚三类库、观察、扫描游标、经验、已见医案和门控分配记录。已落盘计划保留供同一提案重试。
- 冻结评测只用固定 eval_view=integrated 的单输出，不按测试原方挑最优候选；原方仅在生成后评分。
- 旧 v1/v2/v3/v4 可以只读诊断及回放练习候选，不能在旧运行名上强行续跑 v5。旧提案和技能不因本次规则变化被追认接受，也不自动迁入假设库；新协议重新归纳。
- 调用日志、原始解释、评分向量、definition_review、support_checks、skill_routing、gate_plans、来源签名保留，便于人工复核。

## 10. 特别需要人工确认的设计选择

这些是当前实现，不是自然定律：min_gap=0.05、min_support=3、top-k=4、128 组单次预筛预算、12 对假设复查/步、1 条待验证尝试/步、ΔF1=0.005、t=1.28、按组累计的 guard、未知排除项的处理、procedure/medical 和对比支持的 LLM 分类。
尚未做专家校准、引用语义蕴含审计、近重复语义分组、多随机种子稳定性、描述组 bootstrap 或多重检验修正。知识来源是仓库可追溯来源，不声称已核实上游医学权威性。

当前代码没有降低接受阈值，也没有为了让 rules>0 跳过支持检查或门控。下一步可以调整上述研究设计，但应另建协议并保留对照。
