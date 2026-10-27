"""Build the GTA-V manifest without starting training.

uv run python -m mars.data.gtav --config configs/datasets/gtav.yaml
uv run python -m mars.data.gtav --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.gtav.config import GtavConfig
from mars.data.gtav.manifest import ensure_manifest


def _gtav_config(raw: dict, config_path: str) -> GtavConfig:
    """Build GtavConfig from a model YAML or configs/datasets/gtav.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return GtavConfig.from_dict(merge(load_corpus_yaml("gtav"), data.overrides))
    if config_path.endswith("gtav.yaml") or raw.get("dataset") == "gtav":
        return GtavConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/gtav.yaml or a model YAML")


def main() -> int:
    """Scan GTA-V clips and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the GTA-V manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/gtav.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_gtav_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
