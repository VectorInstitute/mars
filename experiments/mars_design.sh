#!/bin/bash
# MARS design ablations, seed 0, each against the main-table mars_s0.
#   bash experiments/mars_design.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

submit mars 3 ablation_paths mars_residual_only_s0 "model.paths=[residual]"
submit mars 3 ablation_paths mars_semantic_only_s0 "model.paths=[semantic]"

submit mars 3 ablation_streams mars_no_appearance_s0 model.streams.appearance=false
submit mars 3 ablation_streams mars_no_spatial_s0 model.streams.spatial.enabled=false
submit mars 3 ablation_streams mars_no_temporal_s0 model.streams.temporal.enabled=false
submit mars 3 ablation_streams mars_no_first_difference_s0 model.streams.first_difference=false

submit mars 3 ablation_norms mars_frozen_norms_s0 model.tune_semantic_norms=false

for layer in 6 18 24; do
  submit mars 3 ablation_split "mars_split${layer}_s0" "model.encoder_layer=${layer}"
done

for frames in 8 32 64; do
  submit mars 3 semantic_frames "mars_sem${frames}_s0" "model.semantic_frames=${frames}"
done

CONFIG=configs/models/mars/semantic_lora.yaml submit mars 3 semantic_lora mars_lora16_s0

CONFIG=configs/models/mars/ablation_backbone_siglip.yaml submit mars 3 ablation_backbone mars_siglip_s0
CONFIG=configs/models/mars/ablation_backbone_dinov3.yaml submit mars 3 ablation_backbone mars_dinov3_s0
