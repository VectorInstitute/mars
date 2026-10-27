#!/bin/bash
# Robustness: each method's main checkpoint (seed 0) scores degraded copies of the 7 test
# sets (group robustness). No training; the source threshold of the main run is reused.
#   bash experiments/robustness.sh [models...]
# DRY_RUN=1 prints the sbatch lines instead of submitting them.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
models_from_args "$@"
TEST_SETS="[ffpp,celebdf,dftimit,uadfv,hififace_paired,genvideo,df26]"
SETTINGS=(
  jpeg:80 jpeg:60 jpeg:40 jpeg:20 jpeg:10
  blur:0.5 blur:1.0 blur:1.5 blur:2.0 blur:3.0
  resize:224 resize:112 resize:64
  noise:0.01 noise:0.03 noise:0.05 noise:0.1 noise:0.2
)
# Per-GPU evaluation batch where the wrapper's default (16 clips) does not fit in memory;
# it changes speed only, not the scores.
declare -A EVAL_BATCH=([demamba]=2)
for model in "${MODELS[@]}"; do
  run="${model}/main_ffpp_gtav/${model}_s0"
  for setting in "${SETTINGS[@]}"; do
    kind="${setting%%:*}" level="${setting##*:}"
    cmd=(sbatch --time=12:00:00 scripts/eval.sh
      --config "configs/models/${model}/config.yaml"
      --checkpoint "checkpoints/${run}/best.pt"
      --source-threshold "results/${run}/source_threshold.json"
      --experiment "${model}_${kind}${level}_s0"
      --set log.group=robustness "data.test=${TEST_SETS}" "data.perturb=${kind}" "data.perturb_level=${level}")
    [[ -n "${EVAL_BATCH[$model]:-}" ]] && cmd+=("train.eval_batch_size=${EVAL_BATCH[$model]}")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
  done
done
