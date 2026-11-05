"""Build a HifiFace manifest without starting evaluation.

python -m mars.data.hififace --config configs/datasets/hififace_paired.yaml
python -m mars.data.hififace --config configs/datasets/hififace_onlyfake.yaml
"""

from __future__ import annotations

import argparse

from mars.config import load_yaml, parse_overrides
from mars.data.hififace.config import HififaceConfig
from mars.data.hififace.manifest import ensure_manifest


def _hififace_config(raw: dict, config_path: str) -> HififaceConfig:
    """Build HififaceConfig from a corpus YAML."""
    slug = raw.get("dataset")
    if slug in ("hififace_paired", "hififace_onlyfake") or config_path.endswith(
        ("hififace_paired.yaml", "hififace_onlyfake.yaml")
    ):
        return HififaceConfig.from_dict(raw)
    raise ValueError("pass configs/datasets/hififace_paired.yaml or hififace_onlyfake.yaml")


def main() -> int:
    """Scan HifiFace videos and write the corpus manifest CSV."""
    parser = argparse.ArgumentParser(description="Build a HifiFace manifest CSV")
    parser.add_argument(
        "--config",
        required=True,
        help="configs/datasets/hififace_paired.yaml or hififace_onlyfake.yaml",
    )
    parser.add_argument("--set", nargs="+", action="extend", default=[], help="dotted key=value overrides")
    args = parser.parse_args()
    raw = load_yaml(args.config, parse_overrides(args.set))
    path = ensure_manifest(_hififace_config(raw, args.config), is_main=True)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
