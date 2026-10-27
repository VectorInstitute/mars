# GenD

Re-implementation of GenD under the shared protocol:

> Andrii Yermakov, Jan Cech, Jiri Matas, Mario Fritz. **Deepfake Detection that Generalizes
> Across Benchmarks.** WACV 2026. [arXiv:2508.06248](https://arxiv.org/abs/2508.06248),
> code: [yermandy/GenD](https://github.com/yermandy/GenD).

```bibtex
@inproceedings{GenD,
    title     = {Deepfake detection that generalizes across benchmarks},
    author    = {Yermakov, Andrii and Cech, Jan and Matas, Jiri and Fritz, Mario},
    year      = 2026,
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
    pages     = {773--783}
}
```

## What is taken from the paper

| part | GenD (PE-L, full model) |
|---|---|
| encoder | Perception Encoder `vit_pe_core_large_patch14_336` (timm) |
| trainable | every encoder parameter whose name contains `norm`, plus the head |
| head | L2-normalised pooled feature → linear |
| loss | CE + 0.5 × uniformity + 0.1 × label-aware alignment on the unit embeddings |

## What follows our protocol instead

- Input: 64-frame clips at the encoder's native 336 px, no face crop, no augmentation.
- Frames: every genuine frame is encoded and back-propagated, one sample each with its video's
  label; the clip logit is the mean of frame logits over genuine frames. Padded frames are never
  encoded.
- Recipe, seed, selection, metrics: shared (`train/`, `eval/`).

## Files

```
models/gend/
├── config.py     # ModelConfig (encoder), LossConfig (term weights)
├── encoder.py    # Perception Encoder (timm) with block-level access
├── model.py      # GenD network: tuned norms, L2 head per frame, mean over genuine frames
├── objective.py  # per-frame CE, uniformity, label-aware alignment
└── detector.py   # BaseDetector: network and objective
```
