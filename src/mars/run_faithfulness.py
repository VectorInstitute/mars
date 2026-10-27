"""Faithfulness tests for trained MARS.

uv run python -m mars.run_faithfulness \\
        --config configs/models/mars/config.yaml \\
        --checkpoint checkpoints/mars/main_ffpp_gtav/mars_s0/best.pt --test splice
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mars.config import load_yaml, parse_overrides
from mars.eval.faithfulness.run import DELETION_CORPORA, run_deletion, run_splice, run_temporal
from mars.models.factory import build_detector


def main() -> int:
    """Parse CLI args and run the requested faithfulness test."""
    parser = argparse.ArgumentParser(description="Faithfulness tests for MARS")
    parser.add_argument("--config", required=True, help="Path to the MARS YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to a checkpoint .pt")
    parser.add_argument("--test", required=True, choices=["splice", "deletion", "temporal"])
    parser.add_argument(
        "--main-run",
        default="results/mars/main_ffpp_gtav/mars_s0",
        help="Main run results (deletion picks clips from it)",
    )
    parser.add_argument(
        "--corpora", nargs="+", default=list(DELETION_CORPORA), help="deletion: test sets to use (default all 7)"
    )
    parser.add_argument("--out", default=None, help="Output folder (default results/mars/faithfulness/<test>)")
    parser.add_argument(
        "--limit", type=int, default=None, help="splice: only the first n pairs; deletion: n clips per class and corpus"
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()

    raw = load_yaml(args.config, parse_overrides(args.set))
    detector = build_detector(str(raw["model"]["name"]), raw)
    out = Path(args.out or f"results/mars/faithfulness/{args.test}")
    if args.test == "splice":
        run_splice(detector, args.checkpoint, out, limit=args.limit)
    elif args.test == "temporal":
        run_temporal(detector, args.checkpoint, out, limit=args.limit)
    else:
        run_deletion(
            detector, args.checkpoint, out, Path(args.main_run), tuple(args.corpora), n_per_class=args.limit or 100
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
