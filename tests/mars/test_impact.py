"""Footprint reports: main-rank measuring, run and per-corpus impact.json, lineage."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mars.models.factory import DETECTORS
from mars.utils.impact import Footprint


@pytest.fixture(autouse=True)
def estimate_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the hardware estimate, so tests neither sample power nor start CodeCarbon."""
    monkeypatch.setenv("CODECARBON_DISABLED", "1")


def footprint(is_main: bool) -> Footprint:
    """Build a footprint for one rank."""
    ctx: Any = SimpleNamespace(is_main=is_main)
    return Footprint("openai/clip-vit-large-patch14", "adapter", ctx)


def test_other_ranks_measure_and_write_nothing(tmp_path: Path) -> None:
    fp = footprint(is_main=False)
    fp.start("train")
    fp.stop("train")
    fp.write(tmp_path)
    assert fp.metrics() == {}
    assert not (tmp_path / "impact.json").exists()


def measure(fp: Footprint, *stages: str) -> None:
    """Start and stop each stage in turn."""
    for stage in stages:
        fp.start(stage)
        fp.stop(stage)


def test_each_corpus_gets_its_own_report(tmp_path: Path) -> None:
    fp = footprint(is_main=True)
    measure(fp, "train", "source_val", "eval/celebdf", "eval/vifbench")
    fp.write(tmp_path)
    run = json.loads((tmp_path / "impact.json").read_text())
    assert set(run) == {"train", "source_val"}
    report = run["train"]["dia_report"]
    assert report["lineage"] == [{"model": "openai/clip-vit-large-patch14", "relation": "adapter"}]
    assert {"energy_kwh", "carbon_kgco2eq", "water_liters"} <= set(report["footprint"])
    for corpus in ("celebdf", "vifbench"):
        corpus_report = json.loads((tmp_path / corpus / "impact.json").read_text())
        assert {"energy_kwh", "carbon_kgco2eq", "water_liters"} <= set(corpus_report["dia_report"]["footprint"])
    assert {"impact/train/energy_kwh", "impact/source_val/gpu_hours", "impact/eval/celebdf/energy_kwh"} <= set(
        fp.metrics()
    )


def test_a_later_eval_keeps_the_training_report(tmp_path: Path) -> None:
    (tmp_path / "impact.json").write_text(json.dumps({"train": {"dia_report": {"kept": True}}}))
    fp = footprint(is_main=True)
    measure(fp, "source_val", "eval/vifbench")
    fp.write(tmp_path)
    run = json.loads((tmp_path / "impact.json").read_text())
    assert run["train"] == {"dia_report": {"kept": True}}
    assert "source_val" in run
    assert (tmp_path / "vifbench" / "impact.json").exists()


def test_every_detector_names_its_base_model() -> None:
    from importlib import import_module

    for name, target in DETECTORS.items():
        module, _, cls = target.partition(":")
        detector = getattr(import_module(module), cls)
        assert detector.base_model, name
        assert detector.relation in {"adapter", "lora", "finetune"}, name
