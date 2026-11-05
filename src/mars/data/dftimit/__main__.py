"""Build the DeepfakeTIMIT manifest without starting evaluation.

uv run python -m mars.data.dftimit --config configs/datasets/dftimit.yaml
uv run python -m mars.data.dftimit --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.dftimit.config import DftimitConfig
from mars.data.dftimit.manifest import ensure_manifest


def _dftimit_config(raw: dict, config_path: str) -> DftimitConfig:
    """Build DftimitConfig from a model YAML or configs/datasets/dftimit.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return DftimitConfig.from_dict(merge(load_corpus_yaml("dftimit"), data.overrides))
    if config_path.endswith("dftimit.yaml") or raw.get("dataset") == "dftimit":
        return DftimitConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/dftimit.yaml or a model YAML")


def main() -> int:
    """Scan DeepfakeTIMIT videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the DeepfakeTIMIT manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/dftimit.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_dftimit_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
