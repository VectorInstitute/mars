"""Build the AVID manifest without starting evaluation.

uv run python -m mars.data.avid --config configs/datasets/avid.yaml
uv run python -m mars.data.avid --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.avid.config import AvidConfig
from mars.data.avid.manifest import ensure_manifest
from mars.data.config import DataConfig


def _avid_config(raw: dict, config_path: str) -> AvidConfig:
    """Build AvidConfig from a model YAML or configs/datasets/avid.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return AvidConfig.from_dict(merge(load_corpus_yaml("avid"), data.overrides))
    if config_path.endswith("avid.yaml") or raw.get("dataset") == "avid":
        return AvidConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/avid.yaml or a model YAML")


def main() -> int:
    """Scan AVID videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the AVID manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/avid.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_avid_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
