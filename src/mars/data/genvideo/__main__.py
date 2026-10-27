"""Build the GenVideo manifest without starting evaluation.

uv run python -m mars.data.genvideo --config configs/datasets/genvideo.yaml
uv run python -m mars.data.genvideo --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.genvideo.config import GenvideoConfig
from mars.data.genvideo.manifest import ensure_manifest


def _genvideo_config(raw: dict, config_path: str) -> GenvideoConfig:
    """Build GenvideoConfig from a model YAML or configs/datasets/genvideo.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return GenvideoConfig.from_dict(merge(load_corpus_yaml("genvideo"), data.overrides))
    if config_path.endswith("genvideo.yaml") or raw.get("dataset") == "genvideo":
        return GenvideoConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/genvideo.yaml or a model YAML")


def main() -> int:
    """Scan GenVideo videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the GenVideo manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/genvideo.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_genvideo_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
