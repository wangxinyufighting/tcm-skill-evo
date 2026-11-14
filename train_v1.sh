(
set -e
cd "/Users/ganning/Documents/project_python/大模型推理/中药推荐/tcm-skill-evo"
export DEEPSEEK_API_KEY=magpie

N=500
RUN="knowledge_v7_auditfix_n${N}_seed42"
DIR="$PWD/runs/$RUN"

python scripts/evolve_knowledge.py train \
  --llm openai_compatible --run_name "$RUN" \
  --set data.seed=42 \
  --set "knowledge.n_practice=${N}" --set knowledge.batch=8 \
  --set knowledge.skill_mode=semantic \
  --set knowledge.learning_mode=cultivate \
  --set 'knowledge.exploration_views=[syndrome, integrated]' \
  --set knowledge.hypothesis_trial_budget=0 \
  --set knowledge.gate_size=16 --set knowledge.confirm_size=16 \
  --set knowledge.gate_candidate_limit=128 \
  --set knowledge.min_support=3 --set knowledge.pending_gate_budget=1 \
  --set knowledge.knowledge_tiebreak=true

if [ "$(jq '.book.rules | length' "$DIR/state.json")" -eq 0 ]; then
  printf '%s\n' '正式技能为 0，暂不执行 500 例技能对比，请先检查训练诊断。'
  exit 0
fi

python scripts/evolve_knowledge.py eval \
  --llm openai_compatible --run_dir "$DIR" \
  --split test --n 500 --empty \
  --out "$DIR/test500_without_skill"

python scripts/evolve_knowledge.py eval \
  --llm openai_compatible --run_dir "$DIR" \
  --split test --n 500 \
  --out "$DIR/test500_with_skill"

jq -n \
  --slurpfile a "$DIR/test500_without_skill/metrics.json" \
  --slurpfile b "$DIR/test500_with_skill/metrics.json" \
  '{n: $a[0].n,
    formal_skills: $b[0].rules,
    without_skill_f1: $a[0].f1,
    with_skill_f1: $b[0].f1,
    delta_f1: ($b[0].f1 - $a[0].f1)}'
)