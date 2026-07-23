"""Tests for manifest media path resolution."""

from pathlib import Path

import pytest

from deepfake_moe.utils.paths import default_audio_dir, resolve_audio_path, resolve_video_path


def test_default_audio_dir() -> None:
    assert default_audio_dir("/data") == str(Path("/data") / "processed" / "audio" / "normalized")
    assert default_audio_dir("") == ""


def test_resolve_video_relative() -> None:
    out = resolve_video_path(
        "test/000000.mp4",
        data_root="/projects/aixpert/deepfake_data",
        video_root="raw/LAV-DF/LAV-DF",
    )
    assert out == "/projects/aixpert/deepfake_data/raw/LAV-DF/LAV-DF/test/000000.mp4"


def test_resolve_video_absolute_passthrough() -> None:
    assert (
        resolve_video_path("/abs/clip.mp4", data_root="/data", video_root="raw/X")
        == "/abs/clip.mp4"
    )


def test_resolve_video_relative_requires_roots() -> None:
    with pytest.raises(ValueError, match="data_root"):
        resolve_video_path("test/a.mp4", data_root="", video_root="raw/X")
    with pytest.raises(ValueError, match="video_root"):
        resolve_video_path("test/a.mp4", data_root="/data", video_root="")


def test_resolve_audio_rewrites_stale_absolute() -> None:
    audio_dir = "/projects/aixpert/deepfake_data/processed/audio/normalized"
    stale = (
        "/fs02/projects/aixpert/users/hafez/deepfake-detection/data/"
        "processed/audio/normalized/lavdf/test/000000.wav"
    )
    assert resolve_audio_path(stale, audio_dir=audio_dir) == f"{audio_dir}/lavdf/test/000000.wav"


def test_resolve_audio_relative_join() -> None:
    audio_dir = "/data/processed/audio/normalized"
    assert resolve_audio_path("lavdf/train/1.wav", audio_dir=audio_dir) == f"{audio_dir}/lavdf/train/1.wav"


def test_resolve_audio_strips_manifest_relative_prefix() -> None:
    audio_dir = "/projects/aixpert/deepfake_data/processed/audio/normalized"
    manifest_path = "audio/normalized/lavdf/train/017033.wav"
    assert (
        resolve_audio_path(manifest_path, audio_dir=audio_dir)
        == f"{audio_dir}/lavdf/train/017033.wav"
    )


def test_resolve_audio_none_and_other_absolute() -> None:
    assert resolve_audio_path(None, audio_dir="/audio") is None
    assert resolve_audio_path("", audio_dir="/audio") is None
    assert resolve_audio_path("/other/place.wav", audio_dir="/audio") == "/other/place.wav"
