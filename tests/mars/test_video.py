"""Frame-directory and video-file I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mars.data.video import count_frames, list_frame_files, read_frames


def _write_jpeg_seq(directory: Path, n: int, *, extra: dict[str, bytes] | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        image = Image.new("RGB", (8, 6), color=(i * 10, 0, 0))
        image.save(directory / f"{i:03d}", format="JPEG")
    for name, payload in (extra or {}).items():
        (directory / name).write_bytes(payload)


def test_list_frame_files_numeric_sort_ignores_non_digit(tmp_path: Path) -> None:
    directory = tmp_path / "si2203"
    directory.mkdir()
    for name in ("10", "2", "1", "sa1.wav", ".dircksum"):
        (directory / name).write_bytes(b"x")
    names = [path.name for path in list_frame_files(directory)]
    assert names == ["1", "2", "10"]


def test_list_frame_files_numeric_stem_with_suffix(tmp_path: Path) -> None:
    directory = tmp_path / "images"
    directory.mkdir()
    for name in ("000002.bmp", "000000.bmp", "000001.png", "camera.yaml"):
        (directory / name).write_bytes(b"x")
    names = [path.name for path in list_frame_files(directory)]
    assert names == ["000000.bmp", "000001.png", "000002.bmp"]


def test_count_and_read_frame_dir(tmp_path: Path) -> None:
    directory = tmp_path / "si2203"
    _write_jpeg_seq(directory, 4, extra={"sa1.wav": b"x"})
    assert count_frames(directory) == 4
    frames = read_frames(directory, [0, 2, 2, -1])
    assert frames.shape == (4, 6, 8, 3)
    assert frames.dtype == np.uint8
    np.testing.assert_array_equal(frames[1], frames[2])
    assert frames[3].sum() == 0


def test_read_empty_frame_dir_raises(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    directory.mkdir()
    (directory / "sa1.wav").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="decoded no frames"):
        read_frames(directory, [0, 1])
    assert count_frames(directory) == 0
