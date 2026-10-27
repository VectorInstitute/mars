"""Build the DF26 manifest without starting evaluation.

uv run python -m mars.data.df26 --config configs/datasets/df26.yaml
uv run python -m mars.data.df26 --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.df26.config import Df26Config
from mars.data.df26.manifest import ensure_manifest


def _df26_config(raw: dict, config_path: str) -> Df26Config:
    """Build Df26Config from a model YAML or configs/datasets/df26.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return Df26Config.from_dict(merge(load_corpus_yaml("df26"), data.overrides))
    if config_path.endswith("df26.yaml") or raw.get("dataset") == "df26":
        return Df26Config.from_dict(raw)
    raise ValueError("pass configs/datasets/df26.yaml or a model YAML")


def main() -> int:
    """Scan DF26 videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the DF26 manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/df26.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_df26_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
