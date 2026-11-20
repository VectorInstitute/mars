"""Shared test fixtures: synthetic videos and results. No models, GPU or network."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import av
import numpy as np
import pytest

from marscope.schema import (
    Agreement,
    ClipScore,
    Credentials,
    FaceSummary,
    ModelScore,
    Prechecks,
    ScoreContext,
    Versions,
    VideoInfo,
    VideoResult,
)


NAMES = {"mars": "MARS", "gend": "GenD", "dfd-hr": "DFD-HR"}


def write_video(path: Path, n_frames: int = 10, width: int = 64, height: int = 48, fps: int = 10) -> Path:
    """Write a small synthetic MP4 whose frames change brightness over time."""
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        for i in range(n_frames):
            img = np.full((height, width, 3), (i * 23) % 256, dtype=np.uint8)
            img[: height // 2, : width // 2] = 255 - img[0, 0]
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


@pytest.fixture(autouse=True)
def no_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests away from any LLM (ignore ``.env`` and the LLM settings)."""
    monkeypatch.setattr("dotenv.load_dotenv", lambda *_a, **_k: False)
    for key in ("MARSCOPE_LLM_BASE_URL", "MARSCOPE_LLM_API_KEY", "MARSCOPE_LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture
def video(tmp_path: Path) -> Path:
    """Return a 10-frame 64 x 48 video."""
    return write_video(tmp_path / "clip.mp4")


def model_score(
    model: str, fake: float, high: bool, measures: dict[str, float] | None = None, precision: str = "fp16"
) -> ModelScore:
    """Return a model score with a chosen video score and context."""
    return ModelScore(
        model=model,
        name=NAMES[model],
        fake=fake,
        real=round(1 - fake, 3),
        clips=[ClipScore(index=0, first_frame=0, last_frame=126, n_valid=64, fake=fake, frame_values=[0.0] * 64)],
        context=ScoreContext(
            real_rank=99.0 if high else 40.0,
            fake_rank=60.0,
            real_range=(0.065, 0.241),
            fake_range=(0.117, 0.628),
            high_fake_score=high,
        ),
        frame_values_kind="contribution" if model == "mars" else "probability",
        device="cuda",
        precision=precision,  # type: ignore[arg-type]
        measures=measures or {},
        seconds=1.0,
    )


@pytest.fixture
def make_result() -> Callable[..., VideoResult]:
    """Build a result from per-model (fake, high) pairs and pre-check values."""

    def build(  # noqa: PLR0917
        scores: dict[str, tuple[float, bool]] | None = None,
        measures: dict[str, dict[str, float]] | None = None,
        faces: FaceSummary | None = None,
        bits_per_pixel: float | None = 0.2,
        credentials: Credentials | None = None,
        scope: list[str] | None = None,
        **extra: Any,
    ) -> VideoResult:
        scores = scores or {"mars": (0.1, False)}
        measures = measures or {}
        models = {m: model_score(m, f, h, measures.get(m)) for m, (f, h) in scores.items()}
        high = [m for m, (_, h) in scores.items() if h]
        return VideoResult(
            video=VideoInfo(
                name="clip.mp4",
                duration_s=5.0,
                width=1280,
                height=720,
                fps=30.0,
                n_frames=150,
                n_kept_frames=75,
                n_clips=2,
            ),
            prechecks=Prechecks(
                in_scope=not scope,
                scope_warnings=scope or [],
                bits_per_pixel=bits_per_pixel,
                faces=faces
                if faces is not None
                else FaceSummary(
                    frames_checked=75,
                    frames_with_face=75,
                    frames_with_several=0,
                    usable_frames=75,
                    median_width_patches=8.0,
                ),
                credentials=credentials or Credentials(status="absent"),
            ),
            models=models,
            agreement=Agreement(
                models=list(models), high_fake=high, disagree=len(models) > 1 and 0 < len(high) < len(models)
            ),
            versions=Versions(package="0", protocol="p", policy="0.1.0", checkpoints=dict.fromkeys(models, "abc")),
            **extra,
        )

    return build
