"""Train a registered detector.

uv run python -m mars.run_train --config configs/models/unite/config.yaml
uv run python -m mars.run_train --config configs/models/unite/config.yaml \\
        --set train.epochs=15 model.spatial_pool=attn log.run_name=ff_ce_attnpool_15ep
"""

from __future__ import annotations

import argparse

from mars.config import load_yaml, parse_overrides
from mars.models.factory import build_detector
from mars.train.trainer import train
from mars.utils.distributed import setup_distributed


def main() -> int:
    """Parse CLI args, build the detector, and run its training loop."""
    parser = argparse.ArgumentParser(description="Train a deepfake detector")
    parser.add_argument("--config", required=True, help="Path to a model YAML")
    parser.add_argument("--model", default=None, help="Override model.name")
    parser.add_argument("--experiment", default=None, help="Override log.run_name")
    parser.add_argument(
        "--set",
        nargs="+",
        action="extend",
        default=[],
        help="dotted key=value overrides",
    )
    args = parser.parse_args()

    raw = load_yaml(args.config, parse_overrides(args.set))
    if args.model:
        raw.setdefault("model", {})["name"] = args.model
    if args.experiment:
        raw.setdefault("log", {})["run_name"] = args.experiment

    name = raw.get("model", {}).get("name")
    if not name:
        raise ValueError("model.name is required (set it in YAML or pass --model)")

    ctx = setup_distributed()
    try:
        train(build_detector(str(name), raw), ctx)
    finally:
        ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
