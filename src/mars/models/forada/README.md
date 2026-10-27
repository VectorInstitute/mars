# ForAda

Re-implementation of the Forensics Adapter under the shared protocol:

> Xinjie Cui, Yuezun Li, Ao Luo, Jiaran Zhou, Junyu Dong. **Forensics Adapter: Adapting CLIP for
> Generalizable Face Forgery Detection.** CVPR 2025. [arXiv:2411.19715](https://arxiv.org/abs/2411.19715),
> code: [OUC-VAS/ForensicsAdapter](https://github.com/OUC-VAS/ForensicsAdapter).

```bibtex
@inproceedings{cui2025forensics,
    title     = {Forensics Adapter: Adapting CLIP for Generalizable Face Forgery Detection},
    author    = {Cui, Xinjie and Li, Yuezun and Luo, Ao and Zhou, Jiaran and Dong, Junyu},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year      = {2025}
}
```

## What is taken from the paper

| part | ForAda |
|---|---|
| encoder | OpenAI CLIP ViT-L/14 at 224, frozen |
| adapter | first 8 blocks of a randomly initialised ViT-Tiny over the 256 px frame, 128 learnable queries, CLIP patch tokens fused after adapter blocks 1, 2, 3 (CLIP blocks 1, 8, 15) |
| guided pass | CLIP's blocks rerun with the queries, which attend to CLIP patches through the adapter's per-head bias; a trainable bottleneck adds 0.05 of its output after CLIP block 7 |
| head | unit query tokens → 1D convolutions over queries → MLP → logits |

## What differs, and why

- Loss: cross-entropy only. ForAda's blending-boundary regression and patch-level contrastive
  terms need face-manipulation masks; every method here trains on video labels only. With no
  masks, the released code also trains on cross-entropy alone. The boundary branch, used only by
  those terms, is left out.
- The fusion layer from CLIP into the adapter is a registered, trained module. In the released
  code it is constructed inside the forward pass, so it is re-drawn at random on every call and
  never trained.
- Input and protocol: 64-frame clips at 256 px, full frames, no face crop, no augmentation; every
  genuine frame is back-propagated, one sample each with its video's label, and the clip logit is
  the mean of frame logits over genuine frames. Recipe, seed, selection and metrics are shared.

## Files

```
models/forada/
├── config.py     # ModelConfig: CLIP, adapter, fusion map
├── clip.py       # OpenAI CLIP ViT whose attention takes a bias; loads the released weights
├── adapter.py    # ViT-Tiny adapter, CLIP fusion, per-head bias decoder
├── model.py      # ForAda network: guided CLIP pass, query head, mean over genuine frames
├── objective.py  # per-frame cross-entropy
└── detector.py   # BaseDetector: network and objective
```
