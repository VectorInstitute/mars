"""Build the Celeb-DF v2 manifest without starting evaluation.

uv run python -m mars.data.celebdf --config configs/datasets/celebdf.yaml
uv run python -m mars.data.celebdf --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.celebdf.config import CelebdfConfig
from mars.data.celebdf.manifest import ensure_manifest
from mars.data.config import DataConfig


def _celebdf_config(raw: dict, config_path: str) -> CelebdfConfig:
    """Build :class:`CelebdfConfig` from a model or corpus YAML."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return CelebdfConfig.from_dict(merge(load_corpus_yaml("celebdf"), data.overrides))
    if config_path.endswith("celebdf.yaml") or raw.get("dataset") == "celebdf":
        return CelebdfConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/celebdf.yaml or a model YAML")


def main() -> int:
    """Read the official Celeb-DF test list and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the Celeb-DF v2 manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/celebdf.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_celebdf_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
