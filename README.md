# TCM-SkillEvo：从医案中自我进化的中药开方技能记忆

新增实验入口：[`scripts/evolve_knowledge.py`](scripts/evolve_knowledge.py)。使用**原始主治全文、多视角知识检索、药方及生成时依据、多维评分、跨批次经验及独立门控**；支持通过 `--reuse_run` 复用历史候选，旧流程保持不变。当前使用方式见 [v7 多方案对比与技能迭代](docs/knowledge_exploration_v7.md)。

v7 的主线是“多方案 → 评分 → 同病例优劣对比 → 总结/修订技能 → 后续批次继续使用”：工作技能直接进入部分常规探索候选，不等待支持票、配对试用或门控。默认 4 个知识视角中的 syndrome/integrated 使用适用的工作技能，其余保留正式库基线；仍按批次更新。三类库表示成熟程度；配对试用默认关闭，独立门控只负责正式晋升。F1 相同时，已知知识支持指标的明确优势也可产生偏好对。v5 的 [流程](docs/knowledge_evolution.md)、[判断清单](docs/knowledge_judgment_rules.md) 及 [v6 配对试用版](docs/knowledge_cultivation_v6.md) 保留为历史说明。

源源不断地给 agent 喂"主治描述 + 原方"，让它像跟师抄方一样：**先自己开方 → 揭晓原方 → 对照反思 → 攒够一批再归纳成技能**，最后在测试集上用 agent 开的方子与原方比对评估。参考并扩展了 SkeMex（*Experience Makes Skillful*, arXiv 2606.09365）的 Read–Write–Assess–Govern 闭环；LLM 权重不变，只更新技能库。默认骨干：DeepSeek（`deepseek-flash`）。

```
每个病例：  检索技能 → 开方（理→法→方→药，每味药标注来源技能）→ 揭晓原方 → 药物级效用更新
每 W 例：   门控 → 复盘反思 → 反思直接生成候选加减规则
            → 相似医案聚类 + 历史近邻 → 多案对比归纳（LLM）
            → 统计检验（LLM 提假设、数据做检验）→ 去重/修订/入库
每 G 窗：   治理：淘汰 / 晋升 / 合并（异病同治）/ 拆分（同病异治）/ 容量
测试：      冻结技能库 → 开方 → 与原方比对
```

## 目录

```
configs/default.yaml     所有超参（命令行 --set a.b=c 可覆盖）
configs/methods.yaml     方法预设：基线 / 本方法 / 消融
kb/                      知识库（别名表、十八反十九畏；清单见 kb/README.md）
scripts/                 每一步实验一个脚本（见"分步实验"）
src/tcmskill/
  data.py        PTM 对齐、描述清洗、GT 重新切分、划分、类别、近重复检测、数据报告
  herbs.py       药名词表/别名/最长匹配切分/LLM 输出归一化、配伍禁忌、功效分类
  llm.py         DeepSeek 客户端（缓存/重试/降级/JSON 修复/计费）     mock.py  模拟后端
  prompts.py     全部提示词（中文）
  skills.py      理法方药结构化技能、技能库、触发匹配、符号执行、检索
  prescriber.py  开方 agent（解析带来源标注的处方）
  learning.py    经验记录、药物级效用归因、门控、反思
  induce.py      聚类、多案对比归纳、统计检验、反思→规则、写入
  govern.py      治理（淘汰/晋升/合并/拆分/容量）
  stats.py       已见病例统计库（IDF、规则支持度/提升度、近邻）
  methods.py     本方法 + 各 LLM 基线（同一骨干、同一输出格式）
  baselines.py   非 LLM 基线（高频、kNN、MLP）
  metrics.py     P/R/F1、P@K、NDCG、soft F1、bootstrap 置信区间与配对检验
  runner.py      流式建库（先预测后更新）/ 冻结测试 / 断点续跑
tests/                   单元测试 + mock 端到端测试
```

## 与技术路线的对应

| 技术路线中的设计                                          | 实现位置                          | 消融开关（methods.yaml）                    |
| --------------------------------------------------------- | --------------------------------- | ------------------------------------------- |
| 先开方、后揭晓的学徒式协议（prequential）                 | `runner.run_stream`             | `abl_hindsight`（不先开方，只看原方归纳） |
| 理法方药结构化、可执行技能（三分支：方证 / 药症 / 原则）  | `skills.py`                     | `abl_freetext`（自由文本技能）            |
| 药物级可归因效用（+ "检索到但没采纳"的反事实信号）        | `learning.CreditAssigner`       | `abl_episode_credit`（整例效用）          |
| 多案对比归纳（共性→方证，差异→加减）                    | `induce.Clusterer/Inducer`      | `abl_single_case`                         |
| LLM 提假设、数据做检验（支持度/提升度/核心药精度）        | `induce.Verifier`               | `abl_no_verify`                           |
| 反思→加减规则（零额外调用）                              | `induce.rules_from_reflections` | `abl_no_rules`                            |
| 治理：合并（异病同治，多触发）/ 拆分（同病异治）          | `govern.py`                     | `abl_no_govern`、`abl_no_split`         |
| 技能 + 病例混合                                           | `methods.SkillMethod`           | `ours_hybrid`                             |
| SkeMex 式复现（自由文本三分支、整例效用、无检验、无拆分） | 同一框架                          | `skemex_like`                             |

## 安装

```bash
python -m pip install -r requirements.txt     # Python ≥ 3.10；openai / json_repair 可选（没有 openai 会自动用 urllib）
export DEEPSEEK_API_KEY=sk-...                # 只有真实调用时需要；mock 模式不需要
bash scripts/download_ptm.sh                  # 克隆 PTM 到 data/raw/PTM（无法访问 GitHub 时手动下载放到同一位置）
```

**DeepSeek 模型名（2026-11）**：`deepseek-chat` / `deepseek-reasoner` 已于 2026-07-24 下线；当前用 `deepseek-flash`（V4.1-Flash）或 `deepseek-v4-pro`。思考模式是请求参数（`--set llm.thinking=true`，强度 `llm.reasoning_effort=low|high|max`）。价格写在 `configs/default.yaml` 的 `llm.prices`，请以官方价格页为准；2026-08 起有峰谷价（谷时半价），大批量实验建议错峰跑。

## 知识库

完整清单见 [`kb/README.md`](kb/README.md)。概括：PTM 数据与其自带的功效分类、症状–证候表由脚本自动下载；别名表和十八反十九畏我已整理好放在 `kb/`；**教材方剂表（仅用于"重发现率"分析）和外部临床数据集需要你自己准备**。

## 数据中的坑及处理（`prepare_data.py` 自动完成，数字来自数据报告）

| 问题                         | 现象                                                                                                            | 处理                                                                                                                                                                                         |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PTM 预处理用子串匹配         | "白附子"被拆出"附子"、"桑根白皮"只剩"桑"、"罂粟壳"同时给出"罂粟"                                                | 原始 98,334 行与预处理 33,765 行可**逐行精确对齐**（含官方训练/测试划分），在原方文本上用"最长匹配 + 别名表"重新切分；括号里的炮制说明（如"姜汁炒"）不再产生假药。共删除 12,263 个伪影 |
| 古籍药名在词表外             | "黄耆"出现 1,700+ 次但原 GT 完全丢失（黄芪被严重低估），桂心、陈橘皮、生地、丹皮、"芎?"（罕见字被存成 ?）等同理 | 别名表恢复 39,437 个药物出现；同物异名合并（麦门冬→麦冬），炮制品默认并入基原药（`data.merge_processed`）。GT 平均味数 7.15 → 7.73                                                       |
| 症状关键词极稀疏             | 每例平均 1.5 个，70% 只有 1 个（"百损久虚……善惊昏愦……"只抽出"善惊"）                                        | 默认输入**清洗后的完整主治描述**；关键词仍可选（`data.input_mode=keywords`）。kNN 实测：描述 F1 0.357 vs 关键词 0.228                                                                |
| 描述里有出处                 | 3,179 条含《书名》引文，可能泄露方源                                                                            | 清洗去除                                                                                                                                                                                     |
| 训练/测试近重复              | 测试集 994 例（19.8%）与训练集近重复，其中 912 例描述+药物完全相同                                              | 每个测试病例带 `near_dup` 标记；**所有结果同时报告去重子集**。kNN：近重复 F1 0.554，去重 0.313                                                                                       |
| 数据污染（大模型见过经典方） | 无法直接检测                                                                                                    | 报告零样本基线；按"模式常见度"分桶（rare / mid / common）；建议加外部临床集                                                                                                                  |
| 原方本身含配伍禁忌           | 2.5% 的 GT 含十八反/十九畏                                                                                      | 指标里同时给出 GT 自身违规率（`gt_unsafe`）作参照                                                                                                                                          |
| 一症多方，F1 天花板低        | GT 只是合理处方之一                                                                                             | soft F1（同功效类别给半分）+ 专家盲评表                                                                                                                                                      |
| 模型别名背后的版本会变       | `deepseek-flash` 10 月刚升级过                                                                                | 每次调用记录实际模型版本（`usage.json` 的 `models_seen`）；对比实验在同一时间段跑完；响应缓存保证复现                                                                                    |
| 成本                         | —                                                                                                              | sqlite 响应缓存（同一请求只付一次钱，断点续跑免费）、mock 估价、静态前缀命中 DeepSeek 前缀缓存                                                                                               |

想与 PTM 文献数字直接对比时：`python scripts/prepare_data.py --set data.gt_mode=ptm --set paths.processed_dir=data/processed_ptm`，之后所有命令加 `--set paths.processed_dir=data/processed_ptm`（必要时再加 `--set data.input_mode=keywords`）。

## 分步实验

> 默认测试子集是固定随机顺序的前 `--n_test` 例，**所有方法用同一批**；训练流是固定随机顺序的前 `--n_train` 例。
> 结果在 `runs/<方法>_n<训练数>_<顺序>_<模型>/`。中断后用同一条命令重跑即从断点继续（已付费的调用走缓存）。

### Step 1 数据准备（约 20 秒）

```bash
python scripts/prepare_data.py
```

先读 `data/processed/report.md`。

### Step 2 冒烟测试 + 成本估计（不花钱）

```bash
python -m pytest -q tests
python scripts/run_method.py --method ours --llm mock --n_train 2000 --n_test 500
```

mock 按**真实提示长度**和 DeepSeek 价格估算费用（输出长度按经验值估）。我这边估出本方法 2000 建库 + 500 测试约 **$2.4**（`deepseek-flash`）；真实输出通常更长，按 $3–5 预留。全量训练集（2.8 万例）约为其 14 倍；`deepseek-v4-pro` 约 3.3 倍；开思考模式会显著更贵。

### Step 3 非 LLM 基线（几秒）

```bash
python scripts/run_baselines.py --methods freq,knn,knn_kw --n_test 500                 # 记忆 = 全部训练集
python scripts/run_baselines.py --methods knn --n_train 2000 --n_test 500               # 与 LLM 方法同样只"见过"2000 例
python scripts/run_baselines.py --methods mlp --n_test 500                               # 可选，几分钟
```

### Step 4 API 连通性小测（约 $0.01）

```bash
python scripts/run_method.py --method zeroshot_cot --n_test 20 --run_name api_check
```

确认没有报错、`oov`（词表外药名率）接近 0、`usage.json` 里的模型版本正确。

#### 输出预算与完整性检查

所有方法的开方，以及反思、归纳、治理等调用，统一使用 `llm.max_tokens`；开启 `llm.thinking=true` 时，实际请求上限为 `max(llm.max_tokens, llm.thinking_max_tokens)`。不再给 direct / structured 分别写死 400 / 1200 tokens。思考与最终 JSON 共享这个总输出上限；提高上限并不保证端点实际接受或回答质量提升。

主对照 `zeroshot_cot`、`case_rag`、`ours` 保持相同的结构化输出格式。`zeroshot` 保留只输出药名的 direct 格式，是单独的格式对照，不应宣称其提示和 CoT 完全相同。每次运行保存 `output_protocol.json`，记录模型、端点、预算、思考强度、格式与修复策略，比较结果前检查这些条件。

```yaml
llm:
  thinking: true
  reasoning_effort: high
  max_tokens: 4096
  thinking_max_tokens: 16000
  json_repair_retries: 1
  accept_repaired_json: false
```

- `json_repair_retries=1`：遇到截断、无法解析或只能语法修复的输出，最多额外生成一次，沿用同一预算。HTTP 重试仍由 `max_retries` 控制。
- `accept_repaired_json=false`：修复半截 JSON 不再算成功；重试仍失败的病例保留在评测分母，残片药名放在 `rejected_herbs`，不进入计分处方或技能归纳。`finish_reason=length` 即使碰巧返回合法 JSON 也拒绝。
- `llm_calls.jsonl` 保存病例 ID / 请求键、实际预算、最终 `finish_reason`、`json_status`，以及每一轮 `attempts` 的终止原因、用量、错误和参数降级记录。首轮截断不会被重试成功覆盖。`usage.json` 的调用/错误数按调用轮次计，不是病例失败率；其 `output_by_task` 按逻辑请求分别汇总开方、反思、归纳等任务的首轮/最终 JSON 状态与终止原因。JSON 层接受不代表处方非空，病例成功率看预测与评测文件。
- `test_preds.jsonl` / `val_preds.jsonl` 保存同样的输出诊断元数据；`*_output_stats.json` 和 `*_metrics.md` 分别报告完整 JSON、提取的 JSON、语法修复残片、空响应、无法解析、空处方和截断。`json_status=complete` 只表示语法完整，不保证处方有效。
- **JSON 状态与空处方是独立维度**：完整的 `{"herbs":[]}` 同时计入完整 JSON 与空处方。无 JSON 的失败不伪装成空处方；缺失的终止原因记为 `unknown`，不猜成 `stop`。`had_truncation` / `had_repaired_fragment` 包括首轮和重试历史，即使最终成功或复用了缓存也保留。
- 新缓存键包含端点、实际输出预算及输出协议版本，旧协议缓存不会复用；调大 `thinking_max_tokens` 也会产生新键。历史 run 不自动改写或补造诊断信息。

先用新目录做小规模建库和验证集检查，不复用旧运行目录中的建库断点：

```bash
python scripts/run_method.py --method ours --n_train 100 --n_test 30 --test_split val --run_name ours_output_v2_smoke
```

该小跑只检查输出链路，不足以证明方法效果。冻结旧库只重测可用 `--phase test --load_memory <旧库路径>`，但不能修复建库阶段已经丢失的反思。

#### 连接失败与安全重跑

- 默认并发为 `llm.workers=4`。这只是减少请求压力，不保证能修复代理或上游故障；思考预算、提示和技能验证阈值不变。
- 失败预测不进入技能信用更新：不更新效用、检索/使用计数、漏用奖励或类别 F1 基线。正常返回但得分为 0 的有效处方仍参与学习。
- 建库的开方、反思、归纳和治理请求在 HTTP 重试耗尽后会中止本次运行，本窗口不写入建库进度或记忆断点，也不会继续评测。已完成窗口和成功响应缓存保留。`stream_failures.jsonl` 记录中止位置，`usage.json` 在中止时也会保存；网络恢复后重新运行同一命令，从上一完整窗口续跑。门控随机状态随断点保存。
- 评测阶段的请求失败仍留在总病例数中计分，诊断表单列 `请求失败`，不再混入 `未知/非 LLM` 或 `空处方`。新调用的 `attempts[].transport_attempts` 保存每次 HTTP 尝试、底层异常链、HTTP 错误状态和重试等待；JSON 重生成与网络重试分别记录。旧成功缓存仍可复用，但不补造旧缓存缺失的网络历史。
- 修复前的建库断点可能已经受失败请求影响，不能直接续跑；程序会拒绝这类旧断点。不要加载旧技能库或覆盖旧目录，应换新名字重建。`--phase test --load_memory ...` 仅适合旧库诊断，不能补回训练。

保持同一批 500 个训练病例和 300 个验证病例重跑本方法：

```bash
python scripts/run_method.py --method ours \
  --n_train 500 --n_test 300 --test_split val \
  --set llm.workers=4 \
  --run_name ours_think16k_n500_val300_connfix_v1
```

若连接错误再次导致中止，先检查 `llm_calls.jsonl` 中的 `error_chain`，排查网络后再续跑，不要靠放宽技能阈值或持续增大 token 预算绕过故障。

### Step 5 无记忆 LLM 基线与病例检索

```bash
python scripts/run_method.py --method zeroshot --n_test 500
python scripts/run_method.py --method zeroshot_cot --n_test 500
python scripts/run_method.py --method case_rag --n_train 2000 --n_test 500
python scripts/run_method.py --method case_rag --phase test --n_test 500 --set case_rag.memory_scope=all_train --run_name case_rag_alltrain
```

### Step 6 本方法 pilot（约 1 小时，$3–5）

```bash
python scripts/run_method.py --method ours --n_train 2000 --n_test 500
python scripts/plot_curves.py --runs "runs/ours_n2000*" "runs/case_rag_n2000*" --out figures/pilot_curve.png
python scripts/analyze_skills.py --run runs/ours_n2000_random_deepseek-flash
```

重点看：

- `stream_log.jsonl` 每个窗口的 `observe.write`：`created:*`、`rejected:low_precision`、`rejected:too_few_matches`、`rule_not_supported` 的比例。拒绝过多就放宽 `skills.verify_thr`，技能质量差就收紧；
- `analysis/skills.md`：通读一遍学到的技能是否像样；
- 学习曲线（prequential F1）是否随病例数上升。

**调参只在验证集上做**：`--test_split val`（1000 例，从 PTM 训练集划出，不参与建库）。

### Step 7 记忆型 agent 基线

```bash
for m in reflexion expel dc skemex_like; do
  python scripts/run_method.py --method $m --n_train 2000 --n_test 500
done
```

这些是在同一框架、同一骨干、同一输出格式下的**简化复现**（Reflexion / ExpeL / Dynamic Cheatsheet / SkeMex 的核心机制），论文里需注明；SkeMex 官方代码若已开源，建议再跑一遍官方实现。

### Step 8 消融

```bash
for m in ours_hybrid abl_freetext abl_episode_credit abl_no_verify abl_single_case abl_no_rules abl_no_govern abl_no_split abl_hindsight; do
  python scripts/run_method.py --method $m --n_train 2000 --n_test 500
done
```

### Step 9 汇总与显著性

```bash
python scripts/evaluate.py --runs "runs/*"                      # Markdown 总表：F1（全部/去重/近重复/rare/common）、soft F1、味数、配伍禁忌率、费用
python scripts/evaluate.py --compare runs/ours_n2000_random_deepseek-flash runs/case_rag_n2000_random_deepseek-flash   # 配对 bootstrap
```

### Step 10 扩展实验

| 实验                               | 命令要点                                                                                                                                                                        |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 全量建库 / 全量测试                | `--n_train 0 --n_test 0`                                                                                                                                                      |
| 数据量曲线                         | `--n_train 500/1000/2000/5000/...`                                                                                                                                            |
| 遗忘（按类别顺序的数据流）         | `--order category`；对照：`export_sft.py --order category --chunks 5` 做持续微调                                                                                            |
| 跨模型迁移（大模型建库、小模型用） | `--model deepseek-v4-pro` 建库后：`--model deepseek-flash --load_memory runs/<v4pro run>/memory_final.json --run_name transfer_pro2flash`                                   |
| 思考模式                           | `--set llm.thinking=true --set llm.reasoning_effort=high`                                                                                                                     |
| 关键词输入（与 PTM 文献同设定）    | `--set data.input_mode=keywords`（建议同时用 `data.gt_mode=ptm` 的数据目录）                                                                                                |
| 更严格的样本外检验                 | `--set skills.verify_thr.holdout_source=true`                                                                                                                                 |
| 安全过滤                           | `--set safety.mode=filter`（自动删除违规对中排序靠后的药）                                                                                                                    |
| 微调基线（需本地开源模型）         | `python scripts/export_sft.py --out data/sft` → LLaMA-Factory 等训练 → 输出整理成 `{"cid", "text"}` → `python scripts/evaluate.py --pred out.jsonl --out runs/sft_xxx` |
| 专家盲评                           | `python scripts/export_blind_review.py --runs runs/ours_... runs/case_rag_... --n 100 --dedup_only --out review/blind_100`；技能库评审表见 `analysis/skills_review.csv`     |
| 教材方剂重发现率                   | 准备 `kb/formulas.csv` 后：`python scripts/analyze_skills.py --run runs/ours_... --formulas kb/formulas.csv`                                                                |
| 符号执行（不经 LLM 只用技能库）    | `analyze_skills.py` 自动输出 `symbolic_test_*.json`（覆盖率、F1）——说明知识确实沉淀在库里                                                                                 |

### 第 2 阶段：决策层、选方加减、离线技能库（`bash phase2.sh`）

第 1 阶段（val 前 300 例）的结论：全量病例记忆（`case_rag.memory_scope=all_train`）比 500 例记忆高 0.12；
思考模式贵约 6 倍，只在近重复病例上略好，在去重 / rare 子集上反而略差，所以第 2 阶段全部用非思考模式。

| 组件 | 做什么 | 入口 | 费用 |
|---|---|---|---|
| 先例子集 | 评测新增"F1 唯一先例 / F1 无先例"两列：训练集里有描述完全相同且药物完全相同的病例（照抄即可）vs 没有描述相同的病例（真正需要推理） | `evaluate.py`（`--slice 0:300`、`--common`） | 0 |
| 决策层 | 唯一先例 → 直接照抄；其余病例把 LLM 处方与 kNN 投票融合并校准味数。参数只在验证集上调（默认 5 折交叉拟合），测试集用 `--params` 套用 | `scripts/posthoc.py` | 0 |
| 选方加减 | `prescribe.style=select_edit`：候选医案编号 C1..Ck（带相似度），模型先选底方再加减，加药须给原文依据；最终处方 = 底方 − remove + add | 预设 `case_select` | ≈ CaseRAG |
| 离线技能库 | 一次读完整个训练集：方族聚类 → 方证技能（字面触发词、核心药、族内加减）+ 全局"症状→药"加减规则，用与在线流程相同的检验器；可选 LLM 命名 | `scripts/build_skills_offline.py` | 统计 0；命名 ≈ $0.3–1 |
| 医案 → 方族链接 | `skills.link_cases=true`：给出相似医案时，一并给出这些医案所属方族的技能（医案行注明"属方族：F12"） | 预设 `ours_offline` / `ours_offline_select` | ≈ CaseRAG + 技能 |

离线技能库的统计结果（全训练集）：约 2100 个方族 → 约 970 条方证技能 + 430 条加减规则，用时约 15 秒。单独做符号执行
（不调 LLM）时 F1 约 0.18，远低于 kNN 的 0.34：核心药是同族方的交集，而且字面触发词只在约 18% 的"本该属于该方族"的病例上触发。
因此测试时用 `link_cases` 让技能跟着检索到的医案走，而不是只靠触发词。技能能不能在去重、rare、无先例子集上带来增益，
要看 `ours_offline` 对比 `case_rag` 的配对检验（`phase2.sh` 的 2e）。

### 第 3 阶段：学习式决策层（`bash phase3.sh`）

第 2 阶段（val1000，非思考）：CaseRAG 0.394 → 加决策层 0.435（增益全部来自 211 例唯一先例；704 例无先例几乎不变：0.349 → 0.350）；
选方加减 0.426（近重复上好得多，去重 / rare 上略差）→ 加决策层 0.434；离线技能 + 医案 0.360（比 CaseRAG 低 0.034）→ 0.424。

`scripts/learn_decision.py` 把"留哪些药"交给一个从反馈中学到的模型：候选药 = 各 LLM run 的药 ∪ 近邻投票的药 ∪ 唯一先例的药（∪ 技能建议的药），
每味药有检索 / 先例 / 各 LLM run / 技能四组特征，梯度提升（或逻辑回归）预测"是否在原方中"，每例取使期望 F1 最大的前 k 味。三种用法：

| 用法 | 命令要点 | 说明 |
|---|---|---|
| val 上交叉拟合 + 消融 | `--runs A B C --skills mem.json` | 报告"不用 LLM / +各 run / +技能特征"，并给出 LLM、技能各自的边际价值（配对检验） |
| 练习协议 | `--fit_split train --fit_runs P` | 只在训练病例上的练习作答里学，在 val 上评测；`--curve` 给经验曲线 |
| 不用 LLM 的练习 | `--fit_split train --fit_runs --fit_n 10000` | 零费用的强基线：训练流前 N 例（检索时排除自己）上学 |
| 测试集 | `--model model.pkl` | 套用保存的模型，测试集上不训练 |

注意：练习病例不能参与构建技能库（`build_skills_offline.py --skip_first N`），herb 频次特征也不做"排除自己"处理——
两者都会把标签泄漏进特征（脚本会对前者给出警告）。

已知结果（不调 API）：不用 LLM、在 10000 例练习上学到的模型在 val1000 上 F1 0.440（无先例 0.356，rare 0.265），
高于第 2 阶段最好的 LLM 流程（0.435）；练习 500 例时已有 0.432。

### 第 4 阶段：开方技能进化（`bash phase4.sh`）

目标：让 LLM 随着见到的病例增多，自己总结开方规律、形成技能并持续改进；技能的作用用"同一提示词、有 / 无技能"直接对比。
设计来自近两年让经验 / 技能真正起作用的工作：Training-Free GRPO（组内语义优势）、ExpeL（成功 / 失败对比）、
ACE（增量编辑的技能手册）、SkillOpt / GEPA（有界编辑 + 留出集验证）、Trace2Skill（批量归纳）、SkeMex（效用驱动的保留与淘汰）。

一步 = 训练集里的一小批练习病例（检索时排除病例自己）：

1. **组内作答**：a 智能体按当前技能作答（T=0）；b 重新采样（T=0.9）；c 照抄最相似医案 C1；d 多案投票。四种方案都和原方对照打分。
2. **语义优势**：优化器（默认开思考）比较同一病例中高分和低分的方案，归纳可推广的开方规律，提出至多 `k_edit` 条编辑（新增 / 改写 / 删除）。
   技能分通用（总在提示词里）和病证专项（病例描述命中关键词时才出现）。
3. **证据过滤**：每条编辑要在 why 里写“支持 #… ；反例 #…”，支持 ≥ `min_support`（3）个且 ≥ `min_support_ratio`（2）倍反例，否则不检验直接丢弃（只在一两个病例上看到的“规律”多半是过拟合）。交给优化器的病例里，智能体落后的与智能体最好的各占一半（`balanced_reflect`），并附本批与累计练习中各方案按相似度分段的平均表现（`history_table`），避免优化器只看到“照抄 / 投票赢了”的病例而过度推广。
4. **逐条验证门控**：每条编辑单独检验（`gate_mode: per_edit`），相对当前技能在留出的训练病例（`n_sel` 300）上比较：先在前 `n_screen`（100）个受影响病例上初筛，
   平均 ΔF1 ≤ 0 直接拒绝；通过的再看全部受影响病例，配对 ΔF1 ≥ `min_delta` 且 t ≥ `t_min`（1.28）才保留。只改病证专项技能时，再从验证池补命中关键词的病例一起判断。
   被拒绝的编辑（连同各自的 ΔF1）会在后续步骤中提示优化器不要重复。第一版做法（整批检验）保留为 `gate_mode: joint`，用作消融。
5. **复核**：通过门控的编辑再在另外 `n_confirm`（300）个留出训练病例上复核，平均 ΔF1 ≥ `confirm_delta`（0）才保留。同一批门控病例要检验很多条编辑，总有碰巧通过的（winner's curse）；复核集只用于已通过门控的少数编辑，规则记录的增益取复核值。
   优化器开思考，输出上限 `optimizer_max_tokens`（32000，思考内容也占额度）；被截断时自动用 1.5 倍再试一次。

智能体的提示词是中性的，不含任何人工策略（何时照抄、开几味、如何取舍）；这些都必须由进化出的技能提供。
产物：`runs/<run>/skillbook_final.md`（可读的技能文档）、`versions/`（每次被接受的版本）、`evolve_log.jsonl`（每步的对比、编辑和门控结果）。
消融开关：`skillbook.references=false`（不做组内对比）、`skillbook.gate=false`（不做验证门控）、`skillbook.gate_mode=joint`（整批检验）、`skillbook.min_support=0`（不要求多病例支持）、`skillbook.balanced_reflect=false` + `skillbook.history_table=false`（第二版的病例选择）、`skillbook.n_confirm=0`（不复核）。

并发：`configs/default.yaml` 的 `llm.workers`，或每条命令加 `--set llm.workers=32`。训练时每步的作答只有 16 例，并发超过 16 主要加快门控（≤150 例）和评测（1000 例）；
每步还有一次串行的优化器调用（开思考），这是每步耗时的下限。请求在客户端重试耗尽后仍失败的病例不会按 F1=0 计分：先整体重跑 `skillbook.retry_passes` 轮，
仍失败则本步不提交、记入 `evolve_failures.jsonl` 并退出，调低并发或等网络恢复后重跑同一命令续跑。

## 输出文件（每个 run 目录）

| 文件                                                                     | 内容                                                                                        |
| ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------- |
| `config.json`                                                          | 完整配置 + 命令行参数                                                                       |
| `stream_log.jsonl`                                                     | 每个窗口：prequential P/R/F1、平均味数、检索/采用技能数、记忆规模、写入与治理统计、累计费用 |
| `stream_preds.jsonl`                                                   | 建库过程中每例的处方（含每味药来源、辨证、治法）                                            |
| `memory.json` / `memory_final.json` / `memory_wXXXX.json`          | 技能库 + 已见病例 + 效用基线（最新 / 最终 / 每 10 窗快照）                                  |
| `test_preds.jsonl`、`test_per_case.jsonl`、`test_metrics.json/.md` | 测试处方、逐例指标、汇总（全部 / 去重 / 近重复 / 模式常见度分桶）                           |
| `usage.json`、`llm_calls.jsonl`                                      | token、费用、实际模型版本；逐次调用日志                                                     |
| `output_protocol.json`、`test_output_stats.json` / `val_output_stats.json` | 输出条件快照；JSON 完整性、空处方、终止原因与重试诊断                                      |
| `analysis/`                                                            | `analyze_skills.py` 的输出                                                                |

指标说明：`f1/p/r` 为集合指标；`p@5`、`r@10`、`ndcg@10` 按模型输出顺序（君→臣→佐→使）计算；`soft_f1` 对同功效类别的替代药给 0.5 分；`len_pred/len_gt/len_abs_err` 衡量变长与多开；`unsafe`/`gt_unsafe` 为预测/原方的配伍禁忌率；`oov` 为含词表外药名的病例比例；`error` 为病例失败率（包含请求失败、无效输出和空处方），`request_failed` 单列请求失败率。

## 关键超参（`configs/default.yaml`）

- `stream.window`（50）：窗口越大，每个窗口共享同一快照的病例越多，归纳的证据越充分，但更新越慢。
- `skills.verify_thr`：方证技能的最少命中例数、核心药精度、提升度；加减规则的支持度、提升度、置信度。这是决定技能库"质"与"量"的最关键旋钮。
- `skills.cluster`：聚类的药物 Jaccard 阈值、每簇最少/最多例数、历史近邻数、每窗最多归纳调用数。
- `skills.retrieval`：每个分支检索 top-K、最低匹配分、相似度/效用/记忆强度的权重。
- `skills.utility`：效用学习率（余弦预热 0.05→0.2）、有害惩罚、"漏采纳"增益。
- `skills.govern`：淘汰/晋升阈值、合并阈值、拆分所需的最少使用次数与精度差、各分支容量。

## 常见问题

- **报 401/402/404**：key 无效 / 余额不足 / 模型名不对（见上文模型名），程序会立即停止。
- **端点不接受 `thinking` 或 `response_format`**：自动去掉该参数并继续（日志有警告），JSON 解析带一次修复重试。
- **想换别的 OpenAI 兼容模型**（如 Qwen）：`--llm openai_compatible --set llm.base_url=... --set llm.model=... --set llm.api_key_env=...`，并在 `llm.prices` 里加价格。
- **重跑会不会重复花钱**：成功响应缓存在 `runs/llm_cache.sqlite`，相同请求可直接复用；失败、未缓存或提示发生变化的请求仍需重新调用，也可能计费。想强制重新请求就换缓存文件：`--set llm.cache_path=runs/cache2.sqlite`。
- **从头重建**：加 `--fresh`（忽略断点；缓存仍然生效）。
- **扩充药名别名**：在 `test_preds.jsonl` 里统计 `oov` 字段最常见的药名，补到 `kb/herb_aliases.json` 后重跑 `prepare_data.py`。

## 数据与引用

PTM 数据集版权归 China Knowledge Centre for Engineering Sciences and Technology（CKCEST），仅限科研使用。

- Yao L. et al. A Topic Modeling Approach for Traditional Chinese Medicine Prescriptions. IEEE TKDE, 2018.
- Sun H. et al. Experience Makes Skillful: Enabling Generalizable Medical Agent Reasoning via Self-Evolving Skill Memory. arXiv:2606.09365, 2026.
# tcm-skill-evo
