# models

Each detector is a package under this folder. A detector supplies its network and
its objective; `train/` and `eval/` do the rest the same way for every model.

```
models/
├── base.py        # BaseDetector: from_config, build_network, build_objective
├── factory.py     # DETECTORS: model.name -> detector class
├── demamba/       # DeMamba (2024), baseline; see demamba/README.md
├── dfd_hr/        # DFD-HR (CVPR 2026), baseline; see dfd_hr/README.md
├── effort/        # Effort (ICML 2025), baseline; see effort/README.md
├── forada/        # ForAda (CVPR 2025), baseline; see forada/README.md
├── gend/          # GenD (WACV 2026), baseline; see gend/README.md
├── mars/          # MARS; see mars/README.md
├── restrav/       # ReStraV (NeurIPS 2025), baseline; see restrav/README.md
└── unite/         # UNITE replication, baseline; see unite/README.md
```

Each model package holds its whole architecture (encoders, trunks, pooling), so a
change to one model can never change another. Shared across models is only the
protocol: datasets and loaders (`data/`), the training loop, recipe and selection
(`train/`), scoring, metrics and result writing (`eval/`), CE (`losses/ce.py`), and
`utils/` (DDP, seed, logging, `masked_mean` over genuine frames).

Per model: network, objective, `model.*` YAML under `configs/models/<name>/`, and a
`*Detector`. The network takes frames and the padding mask; whether it scores frames
or the whole clip, and whether it uses the mask, is its own choice.

To add a model:

1. `models/<name>/`: `model.py` (network), `objective.py` (loss), `config.py` (`Config` extending
   `RunConfig` with its `model`), `detector.py` (`BaseDetector`).
2. Register it in `DETECTORS` in `factory.py`.
3. `configs/models/<name>/config.yaml` with `model.*` only; model YAMLs never set `train.*` or
   `data.test` (tested), except `train.micro_batch_size`, a memory setting that gradient
   accumulation keeps from changing the effective batch.

A frame-level network scores each frame and averages with `utils.masking.masked_mean(logits, valid, dim=1)`,
so padded frames do not count.
