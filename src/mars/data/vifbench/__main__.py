"""Build the ViF-Bench manifest without starting evaluation.

uv run python -m mars.data.vifbench --config configs/datasets/vifbench.yaml
uv run python -m mars.data.vifbench --config configs/models/unite/config.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_corpus_yaml, load_yaml, merge, parse_overrides
from mars.data.config import DataConfig
from mars.data.vifbench.config import VifbenchConfig
from mars.data.vifbench.manifest import ensure_manifest


def _vifbench_config(raw: dict, config_path: str) -> VifbenchConfig:
    """Build VifbenchConfig from a model YAML or configs/datasets/vifbench.yaml."""
    if "data" in raw:
        data = DataConfig.from_dict(raw["data"])
        return VifbenchConfig.from_dict(merge(load_corpus_yaml("vifbench"), data.overrides))
    if config_path.endswith("vifbench.yaml") or raw.get("dataset") == "vifbench":
        return VifbenchConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/vifbench.yaml or a model YAML")


def main() -> int:
    """Scan ViF-Bench videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build the ViF-Bench manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="Model YAML or configs/datasets/vifbench.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_vifbench_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
