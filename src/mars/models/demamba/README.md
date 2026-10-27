# DeMamba

Re-implementation of DeMamba (the XCLIP, fully fine-tuned variant) under the shared protocol:

> Haoxing Chen, Yan Hong, Zizheng Huang, Zhuoer Xu, Zhangxuan Gu, Yaohui Li, Jun Lan, Huijia Zhu,
> Jianfu Zhang, Weiqiang Wang, Huaxiong Li. **DeMamba: AI-Generated Video Detection on
> Million-Scale GenVideo Benchmark.** [arXiv:2405.19707](https://arxiv.org/abs/2405.19707),
> code: [chenhaoxing/DeMamba](https://github.com/chenhaoxing/DeMamba).

```bibtex
@article{DeMamba,
    title   = {DeMamba: AI-Generated Video Detection on Million-Scale GenVideo Benchmark},
    author  = {Haoxing Chen and Yan Hong and Zizheng Huang and Zhuoer Xu and Zhangxuan Gu and Yaohui Li
               and Jun Lan and Huijia Zhu and Jianfu Zhang and Weiqiang Wang and Huaxiong Li},
    journal = {arXiv preprint arXiv:2405.19707},
    year    = {2024}
}
```

## What is taken from the paper

| part | DeMamba-XCLIP-FT |
|---|---|
| encoder | X-CLIP B/16 (`microsoft/xclip-base-patch16`), fully fine-tuned, frames at 224 with ImageNet normalisation |
| global branch | pooled class tokens averaged over frames → LayerNorm |
| detail branch | patch grid split into zones (zone side 1 patch); each zone's tokens in a column-wise snake within the frame, frame after frame, through one residual Mamba block (RMSNorm, drop path 0.1) and averaged |
| head | LayerNorm over the concatenated zone features, concatenated with the global feature → one logit |
| loss | binary cross-entropy |

The released code names only a local X-CLIP folder; 768-d features on a 14 x 14 patch grid
identify X-CLIP B/16.

## What differs, and why

- Zones: the paper reads each zone's tokens frame after frame. The released reshape takes the
  frame and zone axes in the wrong order, so each "sequence" is a run of raster-adjacent tokens,
  mostly from one frame. Sequences here follow the paper.
- Second scan: as released, it reads the input with its feature channels reversed, not time, so
  both scans run forward in time. Kept as released; it also means no genuine step reads a later
  padded frame.
- Padding: X-CLIP's cross-frame messages skip padded frames (`xclip.py`), and both branches
  average over genuine frames only.
- The selective scan runs step by step (the released code's sequential mode, equal to its parallel
  scan), in float32.
- Input and protocol: 64-frame clips, full frames, no augmentation; X-CLIP exchanges messages in
  groups of 8 consecutive frames, as pretrained. Recipe, seed, selection and metrics are shared.

## Files

```
models/demamba/
├── config.py     # ModelConfig: X-CLIP, zones, Mamba sizes
├── xclip.py      # X-CLIP ViT; cross-frame messages skip padded frames
├── mamba.py      # RMSNorm, selective scan, Mamba block, residual block with drop path
├── model.py      # DeMamba network: X-CLIP, global branch, zone-wise Mamba detail branch
├── objective.py  # binary cross-entropy as two-logit cross-entropy
└── detector.py   # BaseDetector: network and objective
```
