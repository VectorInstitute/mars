# Unite

Replication of Kundu et al., CVPR 2025: frozen SigLIP, temporal transformer,
clip CE, optional attention-diversity (AD) loss. The network and the AD loss stay
in this package; training and scoring are the shared ones.

```
unite/
  detector.py    # BaseDetector: network and objective
  model.py       # UNITE network
  encoder.py     # frozen SigLIP (or a test stub)
  transformer.py # per-token temporal transformer
  positional.py  # sinusoidal time encoding
  pooling.py     # attention pool, AD attention and pooled features
  config.py      # typed YAML
  objective.py   # CE + attention-diversity loss
```

Config: `configs/models/unite/config.yaml` (`_base_`: shared sampling/train/log).
CE-only is `--set`, not a second YAML:

```bash
sbatch scripts/train.sh --config configs/models/unite/config.yaml \
  --set loss.lambda_ad=0 loss.ad.enabled=false log.run_name=ff_ce
```

`spatial_pool` is `mean` or `attn`. Outputs:
`checkpoints/unite/<run_name>/` and `results/unite/<run_name>/`.
