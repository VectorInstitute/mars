"""Every model runs under the one protocol in configs/shared.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


MODEL_CONFIGS = sorted(Path("configs/models").rglob("*.yaml"))


# Memory only: gradient accumulation keeps the effective batch at train.batch_size.
MEMORY_KEYS = {"micro_batch_size"}


@pytest.mark.parametrize("path", MODEL_CONFIGS, ids=str)
def test_model_config_keeps_the_shared_recipe_and_battery(path: Path) -> None:
    raw = yaml.safe_load(path.read_text()) or {}
    recipe = set(raw.get("train") or {}) - MEMORY_KEYS
    assert not recipe, f"{path} overrides train.{sorted(recipe)}; the recipe lives in configs/shared.yaml"
    assert "test" not in (raw.get("data") or {}), (
        f"{path} overrides data.test; the battery lives in configs/shared.yaml"
    )


def test_there_are_model_configs() -> None:
    assert MODEL_CONFIGS
