# Effort

Re-implementation of Effort under the shared protocol:

> Zhiyuan Yan, Jiangming Wang, Zhendong Wang, Peng Jin, Ke-Yue Zhang, Shen Chen, Taiping Yao,
> Shouhong Ding, Baoyuan Wu, Li Yuan. **Effort: Efficient Orthogonal Modeling for Generalizable
> AI-Generated Image Detection.** ICML 2025. [arXiv:2411.15633](https://arxiv.org/abs/2411.15633),
> code: [YZY-stack/Effort-AIGI-Detection](https://github.com/YZY-stack/Effort-AIGI-Detection).

```bibtex
@article{yan2024effort,
    title   = {Effort: Efficient Orthogonal Modeling for Generalizable AI-Generated Image Detection},
    author  = {Yan, Zhiyuan and Wang, Jiangming and Wang, Zhendong and Jin, Peng and Zhang, Ke-Yue
               and Chen, Shen and Yao, Taiping and Ding, Shouhong and Wu, Baoyuan and Yuan, Li},
    journal = {arXiv preprint arXiv:2411.15633},
    year    = {2024}
}
```

## What is taken from the paper

| part | Effort |
|---|---|
| encoder | OpenAI CLIP ViT-L/14 at 224, frozen |
| adaptation | query, key, value and output projections of all 24 blocks split by SVD: the top n - 1 singular directions frozen, the last one (u, sigma, v) trained; biases frozen |
| head | class token after CLIP's final norm, unprojected → linear |
| loss | CE + 1.0 × mean over the 96 split layers of (0.5 ‖U Uᵀ − I‖ + 0.5 ‖V Vᵀ − I‖ + \| ‖W‖²_F − ‖W₀‖²_F \|) |

The two constraints are those of the paper (orthogonality, singular-value norm). The released
trainers carry them commented out and train cross-entropy alone; they are implemented here as
written there, weight 1.0 on their mean, and `loss.regularisation: 0` gives the released
cross-entropy-only training.

## What follows our protocol instead

- Input: 64-frame clips at 224 px, full frames, no face crop, no augmentation; every genuine frame
  is back-propagated, one sample each with its video's label, and the clip logit is the mean of
  frame logits over genuine frames.
- Recipe, seed, selection, metrics: shared (`train/`, `eval/`).

## Files

```
models/effort/
├── config.py     # ModelConfig (CLIP, residual rank), LossConfig (constraint weight)
├── clip.py       # OpenAI CLIP ViT whose attention takes a bias; loads the released weights
├── svd.py        # SVD-split linear layer and its orthogonality and singular-value constraints
├── model.py      # Effort network: CLIP with split attention projections, linear head per frame
├── objective.py  # per-frame CE plus the weighted constraints
└── detector.py   # BaseDetector: network and objective
```
