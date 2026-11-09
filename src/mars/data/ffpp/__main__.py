"""Build the FF++ manifest without starting training.

uv run python -m mars.data.ffpp --config configs/datasets/ffpp.yaml
uv run python -m mars.data.ffpp --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.ffpp.config import FfppConfig
from mars.data.ffpp.manifest import ensure_manifest


def _ffpp_config(raw: dict, config_path: str) -> FfppConfig:
    """Build :class:`FfppConfig` from a model YAML or ``configs/datasets/ffpp.yaml``."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        if "ffpp" not in data.train:
            raise ValueError("model YAML must include ffpp in data.train")
        return FfppConfig.from_dict(merge(load_corpus_yaml("ffpp"), data.overrides))
    if config_path.endswith("ffpp.yaml") or raw.get("dataset") == "ffpp":
        return FfppConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/ffpp.yaml or a model YAML with data.train=ffpp")


def main() -> int:
    """Scan FF++ videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the FaceForensics++ manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML (data.train=ffpp) or configs/datasets/ffpp.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_ffpp_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
