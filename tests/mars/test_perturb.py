"""Robustness degradations: each does what it should; noise is reproducible."""

from __future__ import annotations

import numpy as np
import pytest

from mars.data.config import DataConfig
from mars.data.perturb import perturb


def frames(seed: int = 0) -> np.ndarray:
    """Return textured frames whose fine detail blur and compression can remove."""
    rng = np.random.default_rng(seed)
    smooth = np.linspace(0, 255, 64)[None, :, None, None] * np.ones((2, 64, 64, 3)) / 2
    return np.clip(smooth + rng.normal(0, 40, size=(2, 64, 64, 3)), 0, 255).astype(np.uint8)


def detail(x: np.ndarray) -> float:
    """High-frequency energy: mean absolute difference between neighbouring pixels."""
    x = x.astype(np.float32)
    return float(np.abs(np.diff(x, axis=1)).mean() + np.abs(np.diff(x, axis=2)).mean())


def test_none_returns_the_frames_unchanged() -> None:
    x = frames()
    assert perturb(x, "none", 0) is x


@pytest.mark.parametrize(("kind", "levels"), [("jpeg", (80, 40, 10)), ("blur", (0.5, 1.0, 3.0))])
def test_jpeg_and_blur_remove_more_detail_as_they_get_stronger(kind: str, levels: tuple[float, ...]) -> None:
    x = frames()
    energy = [detail(perturb(x, kind, level)) for level in levels]
    assert detail(x) > energy[0] > energy[1] > energy[2]
    assert perturb(x, kind, levels[0]).shape == x.shape


def test_resize_sets_the_short_side_and_keeps_the_aspect_ratio() -> None:
    x = np.zeros((1, 40, 80, 3), dtype=np.uint8)
    assert perturb(x, "resize", 20).shape == (1, 20, 40, 3)


def test_noise_has_the_requested_std_and_is_reproducible() -> None:
    x = np.full((4, 64, 64, 3), 128, dtype=np.uint8)
    noisy = perturb(x, "noise", 0.05, seed=3)
    assert abs(float((noisy.astype(np.float32) - 128).std()) / 255 - 0.05) < 0.005
    assert np.array_equal(noisy, perturb(x, "noise", 0.05, seed=3))
    assert not np.array_equal(noisy, perturb(x, "noise", 0.05, seed=4))


@pytest.mark.parametrize(
    ("kind", "level", "message"),
    [("sharpen", 1, "unknown"), ("jpeg", 0, "quality"), ("blur", 0.7, "sigma"), ("noise", -1, "non-negative")],
)
def test_bad_settings_are_rejected(kind: str, level: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        perturb(frames(), kind, level)
    with pytest.raises(ValueError, match=message):
        DataConfig(perturb=kind, perturb_level=level)


def test_dataset_applies_the_perturbation_once_after_decoding(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import torch
    from PIL import Image

    from mars.data.dataset import ClipDataset, clip_seed
    from mars.data.records import VideoRecord
    from mars.data.transforms import preprocess
    from mars.data.video import read_frames

    folder = tmp_path / "video"
    folder.mkdir()
    for index, frame in enumerate(frames(1)[[0, 1, 0, 1]]):
        Image.fromarray(frame).save(folder / f"{index:04d}.png")
    record = VideoRecord("v", str(folder), 1, 4, "test")
    settings = {"n_frames": 4, "frame_stride": 1, "image_size": 32}
    clean = ClipDataset([record], DataConfig(**settings))[0].frames
    noisy = ClipDataset([record], DataConfig(**settings, perturb="noise", perturb_level=0.1))[0]
    decoded = read_frames(str(folder), [0, 1, 2, 3])
    expected = preprocess(
        perturb(decoded, "noise", 0.1, seed=clip_seed(ClipDataset([record], DataConfig(**settings)).clips[0])), 32
    )
    torch.testing.assert_close(noisy.frames, expected)
    assert not torch.equal(noisy.frames, clean)
