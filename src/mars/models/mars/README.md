# MARS

**Multiscale Appearance–Residual Streams for Video Forgery Detection.**

One frozen Perception Encoder (PE-Core-L), read two ways:

- **Residual path:** fixed band-pass streams over the patch grid and the frames, a masked
  temporal transformer and an attention pool, on the frozen block-12 patch tokens of all
  frames.
- **Semantic path:** 16 frames through all 24 encoder blocks with only the LayerNorms
  trainable, the encoder's own attention pool, and an L2-normalised linear head per frame.
  Its LayerNorms in blocks 1–12 are trainable copies, so the residual path's input never
  changes.

A gate mixes the two paths' standardised margins per clip. Both paths use the same frozen
attention and MLP weights; only the LayerNorms of blocks 1–12 exist twice (about 50K
parameters). Trainable parameters: 67.54M.

```
          RESIDUAL PATH                         SEMANTIC PATH
          (all frames, no gradient)             (16 frames, with gradient)

   ┌──────────▼───────────┐                ┌──────────▼───────────┐
   │ patch embed + pos    │  same weights  │ patch embed + pos    │
   ├──────────────────────┤ ◄────────────► ├──────────────────────┤
   │ blocks 1–12          │ same attn/MLP  │ blocks 1–12          │
   │ original LayerNorms  │                │ LayerNorm copies     │
   │ frozen               │                │ trainable            │
   └──────────┬───────────┘                ├──────────────────────┤
              │                            │ blocks 13–24         │
              │ block-12 tokens            │ LayerNorms trainable │
              │                            ├──────────────────────┤
   ┌──────────▼───────────┐                │ final norm + pool    │
   │ band-pass streams    │                └──────────┬───────────┘
   │ temporal transformer │                ┌──────────▼───────────┐
   │ attention pool, head │                │ head per frame       │
   └──────────┬───────────┘                └──────────┬───────────┘
              └────────────────► GATE ◄───────────────┘
```

## Files

```
models/mars/
├── config.py              # typed YAML (ModelConfig, StreamsConfig)
├── model.py               # Mars: encoder passes, both paths, gate; build_model
├── residual_path.py       # ResidualPath: streams + masked temporal trunk (single-path MARS alone)
├── semantic_path.py       # SemanticPath: L2-normalised linear head per frame; frame subset
├── streams.py             # ResidualStreams (fixed band-pass) and StreamFusion
├── bandpass.py            # Gaussian / heat smoothing; a band is a difference of two
├── validity.py            # which frames of a padded clip are genuine
├── encoder.py             # frozen encoders: SigLIP, Perception Encoder, DINOv3, test stub
├── lora.py                # optional LoRA on the semantic blocks
├── transformer.py         # temporal transformer whose attention skips padded frames
├── positional.py          # sinusoidal time encoding, stamped by genuine-frame order
├── pooling.py             # attention pool over patch tokens
├── objective.py           # CE on the fused output and on each path
└── detector.py            # BaseDetector: network, objective, stream diagnostics

configs/models/mars/
├── config.yaml                      # MARS
├── ablation_backbone_siglip.yaml    # MARS on SigLIP-so400m (split after block 13 of 27)
├── ablation_backbone_dinov3.yaml    # MARS on DINOv3-L (split after block 12 of 24)
└── semantic_lora.yaml               # MARS + LoRA r16 in the semantic blocks
```

MARS owns its whole architecture: encoders, temporal trunk and pooling live in this package.

## Run

```bash
sbatch scripts/train.sh --config configs/models/mars/config.yaml --experiment mars_s0
sbatch scripts/eval.sh  --config configs/models/mars/config.yaml --checkpoint <run>/best.pt

# score one path of a trained model
  --set model.output=semantic      # or residual
# the earlier design: semantic path continues the residual path's frozen block-12 tokens
  --set model.semantic_early_norms=false
```

## Forward pass

1. Encoder blocks 1–12 run once on all frames, frozen, without gradient.
2. Residual path on the block-12 patch tokens (24 × 24): per-patch LayerNorm → appearance
   projection plus seven band-pass streams → sum → time encoding → masked temporal
   transformer → mean over valid frames → attention pool over patches → head → ℓ_B.
3. Semantic path on 16 frames spread over the valid ones: patch embedding → blocks 1–12 with
   the LayerNorm copies → blocks 13–24 → final norm and the encoder's attention pool →
   L2-normalise → linear head per frame, averaged → ℓ_A.
4. Each margin (fake minus real logit) is standardised with running statistics; the gate
   g = σ(W·[a, b]) mixes them: ℓ = g·z_A + (1 − g)·z_B.

Pooling time before space in the residual path is deliberate: each patch is summarised over
time, then the pool weighs patches.

## Training

Loss = CE(ℓ) + CE(ℓ_A) + CE(ℓ_B). The gate sees detached path outputs, so CE(ℓ) trains only
the gate and each path learns from its own loss.

| loss | trains |
|---|---|
| CE(ℓ_B) | the residual path |
| CE(ℓ_A) | the semantic head, every LayerNorm the semantic path uses (the copies in blocks 1–12, blocks 13–24, final norm, pool) |
| CE(ℓ) | the gate |

The encoder's attention and MLP weights, patch embedding and the original LayerNorms of
blocks 1–12 never train, so the residual path reads the same features at every step.

## Residual streams

Fixed, no parameters: 3 spatial difference-of-Gaussian bands over the patch grid, 3 temporal
bands over the frames, and a frame difference. Each gets a `LayerNorm` and a `Linear` into
`d_model`; the fusion sums them with the appearance stream. Order is fixed by
`StreamsConfig.residual_names` because `stream_projs[i]` is positional in a checkpoint.

```bash
--set model.streams.spatial.enabled=false        # drop a family
--set model.streams.temporal.scales=[0.5,1.0,2.0] # n scales give n-1 bands
--set model.streams.appearance=false             # residual streams only
--set model.streams.operator=heat                # heat kernels instead of DoG
```

## Padded frames

Short clips repeat their last frame. The band-pass filters, the time encoding, the attention
keys and the temporal mean use only the genuine frames, taken from the sampler's frame
count in training and evaluation alike; the semantic path picks its 16 frames from the
genuine ones too. `model.temporal_validity=false` turns masking off.
