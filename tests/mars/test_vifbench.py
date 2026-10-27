"""ViF-Bench manifest: reals, per-generator fakes, and the shipped corpus YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from mars.data.base import BaseDataset
from mars.data.config import FAKE, REAL
from mars.data.factory import dataset_class
from mars.data.vifbench.config import VifbenchConfig
from mars.data.vifbench.dataset import VifbenchDataset
from mars.data.vifbench.manifest import build


def _write_tree(root: Path) -> None:
    """Two reals; each has a fake of the same name under two generators."""
    for name in ("a.mp4", "b.mp4"):
        for folder in ("Real", "Fake/sora-2", "Fake/kling-v1"):
            (root / folder).mkdir(parents=True, exist_ok=True)
            (root / folder / name).write_bytes(b"")
    (root / "Fake" / "sora-2" / "notes.txt").write_bytes(b"")


def test_factory_dispatches_vifbench() -> None:
    assert dataset_class("vifbench") is VifbenchDataset
    assert issubclass(VifbenchDataset, BaseDataset)


def test_build_labels_reals_and_names_each_generator(tmp_path: Path) -> None:
    _write_tree(tmp_path)
    cfg = VifbenchConfig(root=str(tmp_path), generators=["sora-2", "kling-v1"])
    records = {r.video_id: r for r in build(cfg, lambda _path: 120)}
    assert len(records) == 6
    assert records["real/a.mp4"].label == REAL and records["real/a.mp4"].extra["method"] == "real"
    assert records["sora-2/a.mp4"].label == FAKE and records["sora-2/a.mp4"].extra["method"] == "sora-2"
    assert records["kling-v1/b.mp4"].extra["method"] == "kling-v1"
    assert {r.split for r in records.values()} == {"test"}


def test_shipped_yaml_lists_all_nineteen_generators() -> None:
    raw = yaml.safe_load(Path("configs/datasets/vifbench.yaml").read_text())
    cfg = VifbenchConfig.from_dict(raw)
    assert len(cfg.generators) == len(set(cfg.generators)) == 19
    assert {"sora-2", "kling-v1", "hailuo", "gen4-turbo"} <= set(cfg.generators)
    assert cfg.roles == ["test"]
