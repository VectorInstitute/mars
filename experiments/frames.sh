#!/bin/bash
# Frames per clip 8, 16, 32; 64 is the main-table run (group frames).
#   bash experiments/frames.sh [models...]
# Shorter clips give more clips per video, so an epoch costs more: the time limit
# is sized as this many 64-frame epochs (measured on MARS: 8 frames ~3x, 16 ~1.9x).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
models_from_args "$@"
declare -A EPOCH_COST=([8]=12 [16]=7.5 [32]=4.8)
for model in "${MODELS[@]}"; do
  for frames in 8 16 32; do
    submit "$model" "${EPOCH_COST[$frames]}" frames "${model}_f${frames}_s0" "data.n_frames=${frames}"
  done
done
