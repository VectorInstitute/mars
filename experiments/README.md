# Experiments

One script per experiment group. Each submits one `scripts/train.sh` job per run, with a time
limit sized to the model and budget. Every run follows the shared recipe (FF++ + GTA-V, 64 frames,
3 epochs, lr 1e-4, seed 0) and changes only what its row says.

```bash
DRY_RUN=1 bash experiments/seeds.sh unite gend   # print the sbatch lines
bash experiments/seeds.sh unite gend             # submit
```

## All methods

These take models as arguments (default: all 8: mars, unite, gend, forada, dfd_hr, demamba,
effort, restrav).

| script | group | runs per model | what it tests |
|---|---|---|---|
| `main.sh` | `main_ffpp_gtav` | `<model>_s0` | the main table |
| `ffpp_only.sh` | `main_ffpp` | `<model>_s0`, `data.train=[ffpp]` | training without GTA-V |
| `seeds.sh` | `seeds` | `<model>_s5`, `<model>_s10`; `SEEDS="15 20"` for more | spread over seeds 0, 5, 10, 15, 20 |
| `frames.sh` | `frames` | `<model>_f8_s0`, `_f16_s0`, `_f32_s0` | clip length; 64 frames is the main table |
| `epochs.sh` | `epochs` | `<model>_ep5_s0`, `_ep10_s0`, `_ep20_s0` | training length, with the lr decay stretched so every budget ends at the same lr; 3 epochs is the main table |

## MARS design

`mars_design.sh` takes no arguments; it submits 16 MARS runs, each against the main-table `mars_s0`.

| group | run_name | change | what it tests |
|---|---|---|---|
| `ablation_paths` | `mars_residual_only_s0` | `model.paths=[residual]` | residual path alone |
| | `mars_semantic_only_s0` | `model.paths=[semantic]` | semantic path alone |
| `ablation_streams` | `mars_no_appearance_s0` | `model.streams.appearance=false` | without the appearance stream |
| | `mars_no_spatial_s0` | `model.streams.spatial.enabled=false` | without the spatial band-pass streams |
| | `mars_no_temporal_s0` | `model.streams.temporal.enabled=false` | without the temporal band-pass streams |
| | `mars_no_first_difference_s0` | `model.streams.first_difference=false` | without the frame-difference stream |
| `ablation_norms` | `mars_frozen_norms_s0` | `model.tune_semantic_norms=false` | semantic path fully frozen |
| `ablation_split` | `mars_split6_s0`, `_split18_s0`, `_split24_s0` | `model.encoder_layer=6 / 18 / 24` | where PE is split between the paths (12 in MARS) |
| `semantic_frames` | `mars_sem8_s0`, `_sem32_s0`, `_sem64_s0` | `model.semantic_frames=8 / 32 / 64` | frames read by the semantic path (16 in MARS; the residual path keeps 64) |
| `semantic_lora` | `mars_lora16_s0` | `configs/models/mars/semantic_lora.yaml` | LoRA (rank 16) in the semantic blocks, on top of LayerNorm tuning |
| `ablation_backbone` | `mars_siglip_s0` | `configs/models/mars/ablation_backbone_siglip.yaml` | SigLIP-so400m/14 instead of PE, split after block 13 of 27 |
| | `mars_dinov3_s0` | `configs/models/mars/ablation_backbone_dinov3.yaml` | DINOv3-L/16 instead of PE, split after block 12 of 24 |

`common.sh` holds the model list and per-model hours per epoch; `CONFIG=` points a
run at another YAML.
