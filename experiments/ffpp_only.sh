#!/bin/bash
# Main table trained on FF++ only (group main_ffpp).
#   bash experiments/ffpp_only.sh [models...]
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
models_from_args "$@"
for model in "${MODELS[@]}"; do
  submit "$model" 3 main_ffpp "${model}_s0" "data.train=[ffpp]"
done
