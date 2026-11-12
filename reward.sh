#!/usr/bin/env bash
# =============================================================================
# 第 1 步：便宜的对照 + 诊断（全部在 val 前 300 例上，与报告 A 组同一批病例）
# 用法：在仓库根目录 bash phase1.sh；也可以逐段复制执行。
# 所有 LLM run 都显式给 --run_name（模型名里有斜杠，不给会多嵌一层目录）。
# 已有响应走缓存，中途断了重跑同一条命令即可。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

A1=runs/zeroshot_cot_think16k_n500_val300          # 报告 A1（零样本 CoT，思考）
A2=runs/ours_think16k_n500_val300_connfix_v1        # 报告 A2（本方法，500 例建库，思考）
Q1=runs/case_rag_think16k_n500_val300               # 报告 Q1（CaseRAG，已见 500 例）

# ---------------------------------------------------------------- 1a 非 LLM 基线（几秒，不花钱）
# 与 LLM 同一批 val300；记忆分别为 全部训练集 / 500 例 / 2000 例
# 写到单独的 runs/val300_baselines/，不覆盖你 test500 基线目录里的 config.json
VB=runs/val300_baselines
python scripts/run_baselines.py --methods freq,knn --test_split val --n_test 300 --set paths.runs_dir=$VB
python scripts/run_baselines.py --methods knn --test_split val --n_test 300 --n_train 500 --set paths.runs_dir=$VB
python scripts/run_baselines.py --methods knn --test_split val --n_test 300 --n_train 2000 --set paths.runs_dir=$VB
# SDPR/CLEPR 预训练基线（同一批 val300；需要 torch 与原 CLEPR 项目，不与本项目的 n_train 挂钩）
SDPR_ROOT=${SDPR_PROJECT_ROOT:-/Users/ganning/Documents/project_python/CLEPR_v2}
if python -c 'import torch' >/dev/null 2>&1 && [ -d "$SDPR_ROOT" ]; then
  python scripts/run_sdpr.py --test_split val --n_test 300 --set paths.runs_dir=$VB --run_name sdpr_nall
  # 仅诊断：去掉 SDPR train/valid 已见病例（val300 中只剩 26 例）
  python scripts/run_sdpr.py --test_split val --n_test 300 --eval_subset nonleak --set paths.runs_dir=$VB --run_name sdpr_nall_nonleak --no_save_scores
else
  echo "跳过 SDPR/CLEPR 基线：需要 torch 与 $SDPR_ROOT"
fi

# ---------------------------------------------------------------- 1b 诊断已有的 A2（不花钱）
python scripts/diagnose_run.py --run "$A2" --split val | tee "$A2/diagnose_val.txt"

# ---------------------------------------------------------------- 1c 全量记忆 CaseRAG（思考，约 $1.5/个）
python scripts/run_method.py --method case_rag --phase test --test_split val --n_test 300 \
  --set case_rag.memory_scope=all_train --set prescribe.n_cases=5 \
  --run_name s1_caserag_all_k5_think
python scripts/run_method.py --method case_rag --phase test --test_split val --n_test 300 \
  --set case_rag.memory_scope=all_train --set prescribe.n_cases=10 \
  --run_name s1_caserag_all_k10_think

# ---------------------------------------------------------------- 1d 关闭思考（约 $0.3/个）
python scripts/run_method.py --method zeroshot_cot --phase test --test_split val --n_test 300 \
  --set llm.thinking=false \
  --run_name s1_cot_nothink
python scripts/run_method.py --method case_rag --phase test --test_split val --n_test 300 \
  --set case_rag.memory_scope=all_train --set prescribe.n_cases=5 --set llm.thinking=false \
  --run_name s1_caserag_all_k5_nothink
python scripts/run_method.py --method case_rag --phase test --test_split val --n_test 300 \
  --set case_rag.memory_scope=all_train --set prescribe.n_cases=10 --set llm.thinking=false \
  --run_name s1_caserag_all_k10_nothink

# ---------------------------------------------------------------- 1e 汇总 + 配对检验
python scripts/evaluate.py --split val --runs $VB/freq_nall $VB/knn_nall $VB/knn_n500 $VB/knn_n2000 $VB/sdpr_nall \
  "$A1" "$A2" "$Q1" runs/s1_* | tee runs/s1_summary_val300.md

python scripts/evaluate.py --split val --compare runs/s1_caserag_all_k5_think $VB/knn_nall      # 全量 RAG 能否打过 kNN
python scripts/evaluate.py --split val --compare runs/s1_caserag_all_k5_think "$A2"             # 全量病例 vs 500 例技能
python scripts/evaluate.py --split val --compare runs/s1_caserag_all_k5_think runs/s1_caserag_all_k5_nothink   # 思考值不值
python scripts/evaluate.py --split val --compare "$A1" runs/s1_cot_nothink                     # 思考对零样本的影响
python scripts/evaluate.py --split val --compare $VB/sdpr_nall $VB/knn_nall                    # SDPR vs kNN（注意 SDPR 训练集与本项目 val300 有重叠）
python scripts/evaluate.py --split val --compare "$A2" $VB/sdpr_nall                          # 主方法 ours vs SDPR（同批 val300）
python scripts/evaluate.py --split val --compare $VB/sdpr_nall $VB/sdpr_nall_nonleak           # 仅诊断：是否受 SDPR 已见病例影响

# 再看一下全量 RAG 的药物来源（参考医案 case vs 模型自己 self 的精度）
python scripts/diagnose_run.py --run runs/s1_caserag_all_k5_think --split val | head -20
