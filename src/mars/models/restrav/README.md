# ReStraV

Re-implementation of ReStraV under the shared protocol:

> Christian Internò, Robert Geirhos, Markus Olhofer, Sunny Liu, Barbara Hammer, David Klindt.
> **AI-Generated Video Detection via Perceptual Straightening.** NeurIPS 2025.
> [arXiv:2507.00583](https://arxiv.org/abs/2507.00583), code:
> [ChristianInterno/ReStraV](https://github.com/ChristianInterno/ReStraV).

```bibtex
@inproceedings{interno2025aigenerated,
    title     = {{AI}-Generated Video Detection via Perceptual Straightening},
    author    = {Intern{\`o}, Christian and Geirhos, Robert and Olhofer, Markus and Liu, Sunny
                 and Hammer, Barbara and Klindt, David},
    booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
    year      = {2025},
    url       = {https://openreview.net/forum?id=LsmUgStXby}
}
```

## What is taken from the paper

| part | ReStraV |
|---|---|
| encoder | DINOv2 ViT-S/14 from `torch.hub` (`facebookresearch/dinov2`, `dinov2_vits14`), frozen, 224 px, pixels in [0, 1] without mean and std |
| trajectory | per frame, the normalised class and patch tokens flattened into one point |
| features (21) | first 7 step lengths, first 6 turning angles (degrees), and mean, min, max, variance of all step lengths and of all angles |
| classifier | features standardised, then MLP 21 → 64 → 32 → 1 |
| loss | binary cross-entropy |

The geometry is `dinov2_features.py` (`compute_temporal_geometry`, `moment4`,
`features_from_Z`) with masking added; the MLP is the one in `train.py`. DINOv2 is loaded
through `torch.hub` as released: timm's DINOv2 at 224 differs by up to 3.7 in token values
(positional embeddings are resized differently), which changes the trajectory.

## What differs, and why

- Frames: all genuine frames of a 64-frame clip at stride 2 (about 12 to 15 per second, close
  to the released 24 frames over 2 s); steps and angles that touch a padded frame are left out
  of every statistic.
- Standardisation: the released code computes feature mean and std over the training set
  before training; here they accumulate over every training sample seen and are frozen in
  evaluation.
- Recipe, seed, selection, metrics: shared (`train/`, `eval/`); the released MLP trains with
  Adam at 1e-3 for 10 epochs.

## Files

```
models/restrav/
├── config.py     # ModelConfig: DINOv2 source, feature counts, MLP widths
├── geometry.py   # step lengths, turning angles and their masked statistics
├── model.py      # ReStraV network: DINOv2 trajectory, running standardiser, MLP
├── objective.py  # binary cross-entropy as two-logit cross-entropy
└── detector.py   # BaseDetector: network and objective
```
