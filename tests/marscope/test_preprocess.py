"""Tests for the MARS input protocol on synthetic videos."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from marscope.video.edits import Edits, apply_edits
from marscope.video.preprocess import build_clips, iter_kept_frames, read_clip, read_meta, to_input
from tests.conftest import write_video


def card_reference(rgb: np.ndarray, size: int) -> np.ndarray:
    """Run the model cards' reference preprocessing, copied verbatim."""
    x = rgb.astype(np.float32) / 255.0
    if x.shape[:2] != (size, size):
        channels = [
            Image.fromarray(np.ascontiguousarray(x[..., c])).resize((size, size), Image.BICUBIC) for c in range(3)
        ]
        x = np.clip(np.stack([np.asarray(c) for c in channels], axis=-1), 0.0, 1.0)
    return ((x - 0.5) / 0.5).transpose(2, 0, 1)


def test_to_input_matches_model_card() -> None:
    rgb = np.random.default_rng(0).integers(0, 256, size=(90, 160, 3), dtype=np.uint8)
    for size in (336, 448, 16):
        np.testing.assert_array_equal(to_input(rgb, size), card_reference(rgb, size))


def test_to_input_range_and_dtype() -> None:
    out = to_input(np.full((8, 8, 3), 255, dtype=np.uint8), 8)
    assert out.dtype == np.float32
    assert out.shape == (3, 8, 8)
    assert np.allclose(out, 1.0)


def test_kept_frames_use_stride(video: Path) -> None:
    indices = [i for i, _ in iter_kept_frames(video, 2)]
    assert indices == [0, 2, 4, 6, 8]


def test_clips_pad_by_repeating_last_frame(video: Path) -> None:
    clips = list(build_clips(iter_kept_frames(video, 2), [16], n_frames=4))
    assert [c.n_valid for c in clips] == [4, 1]
    last = clips[1].inputs[16]
    assert last.shape == (4, 3, 16, 16)
    for t in range(1, 4):
        np.testing.assert_array_equal(last[t], last[0])
    assert clips[1].frame_indices == [8]


def test_clips_for_several_sizes_and_callback(video: Path) -> None:
    seen: list[int] = []
    clips = list(build_clips(iter_kept_frames(video, 2), [8, 12], 64, on_frame=lambda i, _rgb: seen.append(i)))
    assert len(clips) == 1
    assert set(clips[0].inputs) == {8, 12}
    assert seen == [0, 2, 4, 6, 8]


def test_read_clip_matches_streamed_clip(tmp_path: Path) -> None:
    path = write_video(tmp_path / "long.mp4", n_frames=20)
    streamed = list(build_clips(iter_kept_frames(path, 2), [16], n_frames=4))
    again = read_clip(path, 1, 16, 4, 2)
    np.testing.assert_array_equal(again.inputs[16], streamed[1].inputs[16])
    with pytest.raises(IndexError):
        read_clip(path, 9, 16, 4, 2)


def test_read_meta(video: Path) -> None:
    meta = read_meta(video)
    assert (meta.width, meta.height) == (64, 48)
    assert meta.fps == pytest.approx(10.0)
    assert meta.size_bytes > 0


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="no frames"):
        list(build_clips(iter([]), [8], 64))


def test_edits_change_frames_and_identity_does_not(video: Path) -> None:
    plain = read_clip(video, 0, 32, 8, 2)
    same = read_clip(video, 0, 32, 8, 2, transform=lambda i, rgb: apply_edits(rgb, Edits(), seed=i))
    blurred = read_clip(video, 0, 32, 8, 2, transform=lambda i, rgb: apply_edits(rgb, Edits(jpeg=20, blur=2), seed=i))
    assert np.array_equal(plain.inputs[32], same.inputs[32])
    assert not np.allclose(plain.inputs[32], blurred.inputs[32])


def test_edits_reject_out_of_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        Edits(jpeg=5)


def test_meta_has_no_rotation_for_a_plain_video(video: Path) -> None:
    from marscope.video.preprocess import read_meta  # noqa: PLC0415

    assert read_meta(video).rotation == 0
