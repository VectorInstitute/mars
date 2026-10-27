"""Build the UADFV manifest without starting evaluation.

uv run python -m mars.data.uadfv --config configs/datasets/uadfv.yaml
uv run python -m mars.data.uadfv --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.uadfv.config import UadfvConfig
from mars.data.uadfv.manifest import ensure_manifest


def _uadfv_config(raw: dict, config_path: str) -> UadfvConfig:
    """Build UadfvConfig from a model YAML or configs/datasets/uadfv.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return UadfvConfig.from_dict(merge(load_corpus_yaml("uadfv"), data.overrides))
    if config_path.endswith("uadfv.yaml") or raw.get("dataset") == "uadfv":
        return UadfvConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/uadfv.yaml or a model YAML")


def main() -> int:
    """Scan UADFV videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the UADFV manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/uadfv.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_uadfv_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
