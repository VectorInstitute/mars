"""Evaluate a registered detector.

uv run python -m mars.run_eval --config configs/models/unite/config.yaml \\
        --checkpoint checkpoints/unite/ff_ce_ad/best.pt --split test
"""

from __future__ import annotations

import argparse

from mars.config import load_yaml, parse_overrides
from mars.eval.evaluator import run_split
from mars.models.factory import build_detector
from mars.utils.distributed import setup_distributed


def main() -> int:
    """Parse CLI args, build the detector, and run evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a deepfake detector")
    parser.add_argument("--config", required=True, help="Path to a model YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt")
    parser.add_argument("--model", default=None, help="Override model.name")
    parser.add_argument("--experiment", default=None, help="Override log.run_name")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument(
        "--dataset",
        default=None,
        help="Evaluate one corpus from data.test (default: all of data.test)",
    )
    parser.add_argument(
        "--source-threshold",
        default=None,
        help="source_threshold.json of an earlier run of this checkpoint; skips the validation pass",
    )
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
        run_split(
            build_detector(str(name), raw),
            ctx,
            split=args.split,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            source_threshold_file=args.source_threshold,
        )
    finally:
        ctx.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
