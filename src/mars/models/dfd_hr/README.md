# DFD-HR

Re-implementation of DFD-HR under the shared protocol:

> Jiamu Sun, Zhiyuan Yan, Ke-Yue Zhang, Taiping Yao, Shouhong Ding. **DFD-HR: Generalizable
> Deepfake Detection via Hierarchical Routing Learning.** CVPR 2026, pp. 13984–13995.
> Code: [Disguiser15/DFD-HR](https://github.com/Disguiser15/DFD-HR).

```bibtex
@InProceedings{Sun_2026_CVPR,
    author    = {Sun, Jiamu and Yan, Zhiyuan and Zhang, Ke-Yue and Yao, Taiping and Ding, Shouhong},
    title     = {DFD-HR: Generalizable Deepfake Detection via Hierarchical Routing Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {13984-13995}
}
```

## What is taken from the paper

| part | DFD-HR |
|---|---|
| encoder | OpenAI CLIP ViT-L/14 at 224, frozen |
| views | a 448 px frame seen as one 224 global view and 2 x 2 tiles of 224 |
| adapters | an MoE adapter (4 experts, all active, bottleneck 0.25) after the attention and the MLP of all 24 blocks |
| routing | in blocks 21 to 23, a layer router may stop a view and a token router passes the top 75% of tokens, re-weighted by their softmaxed scores |
| head | projected global class token and a 4-head query-attention summary of the 4 tile class tokens → linear |
| loss | CE + 0.1 × (1 − Spearman correlation between block-21 token scores and similarity to the global class token) |

Two properties of the released code are kept as they are:

- The ranking term is computed from argsort ranks and so carries no gradient; it is logged but
  does not change the weights.
- The layer routers' exit decision enters the network only through indexing, so they receive no
  gradient and keep their initial weights; they are frozen explicitly.

## What follows our protocol instead

- Input: 64-frame clips at 448 px, full frames, no face crop, no augmentation; every genuine frame
  is back-propagated, one sample each with its video's label, and the clip logit is the mean of
  frame logits over genuine frames.
- Recipe, seed, selection, metrics: shared (`train/`, `eval/`).

## Files

```
models/dfd_hr/
├── config.py     # ModelConfig (CLIP, adapters, routing), LossConfig (ranking weight)
├── clip.py       # OpenAI CLIP ViT whose attention takes a bias; loads the released weights
├── routing.py    # MoE adapter, token and layer routers, Gumbel sigmoid, Spearman correlation
├── model.py      # DFD-HR network: two views, routed CLIP pass, query-attention head
├── objective.py  # per-frame CE plus the ranking term
└── detector.py   # BaseDetector: network and objective
```
