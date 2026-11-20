"""Reference outputs with the real ONNX files: the pipeline matches the research code.

Needs the model files (downloaded on first use), a GPU for speed, and local copies of
two research-only clips under ``$MARSCOPE_DATA_ROOT`` (never committed)::

    MARSCOPE_DATA_ROOT=/path/to/datasets uv run pytest -m integration_test tests
"""

import os
from pathlib import Path

import numpy as np
import pytest

from marscope.models.registry import get_spec
from marscope.models.runner import Runner
from marscope.pipeline import MARScope
from marscope.video.preprocess import build_clips, iter_kept_frames


ROOT = os.environ.get("MARSCOPE_DATA_ROOT")
CELEBDF = "Celeb-DF-V2/Celeb-synthesis/id3_id9_0001.mp4"
GENVIDEO = "GenVideo-Val/Fake/ModelScope/modelscope_16.mp4"

pytestmark = [
    pytest.mark.integration_test,
    pytest.mark.skipif(ROOT is None, reason="set MARSCOPE_DATA_ROOT to the local dataset folder"),
]


def clip_path(relative: str) -> Path:
    """Return a reference clip, skipping if it is missing."""
    path = Path(str(ROOT)) / relative
    if not path.exists():
        pytest.skip(f"missing {path}")
    return path


def first_clip(path: Path, runner: Runner) -> tuple[np.ndarray, int]:
    """Return the first clip of a video, preprocessed for a runner."""
    p = runner.protocol
    clip = next(build_clips(iter_kept_frames(path, p.frame_stride), [p.image_size], p.n_frames))
    return clip.inputs[p.image_size], clip.n_valid


@pytest.fixture(scope="module")
def mars_fp32() -> MARScope:
    """Return a toolkit running MARS in fp32, as the reference outputs were measured."""
    return MARScope(models="mars", precision="fp32", faces=False)


def test_mars_celebdf_first_clip(mars_fp32: MARScope) -> None:
    result = mars_fp32.analyze(clip_path(CELEBDF))
    assert result.models["mars"].clips[0].fake == pytest.approx(0.6694, abs=1e-3)


def test_mars_genvideo(mars_fp32: MARScope) -> None:
    score = mars_fp32.analyze(clip_path(GENVIDEO)).models["mars"]
    assert score.fake == pytest.approx(0.4745, abs=1e-3)
    assert int(np.argmax(score.clips[0].frame_values)) == 13


def test_fp16_close_to_fp32(mars_fp32: MARScope) -> None:
    fp16 = MARScope(models="mars", precision="fp16", faces=False)
    path = clip_path(GENVIDEO)
    a = mars_fp32.analyze(path).models["mars"]
    b = fp16.analyze(path).models["mars"]
    if b.device == "cpu":
        pytest.skip("fp16 runs only on the GPU")
    assert abs(a.fake - b.fake) <= 0.004


@pytest.mark.parametrize("model", ["mars", "gend", "dfd-hr"])
def test_output_identities(model: str) -> None:
    runner = Runner(get_spec(model))
    frames, n_valid = first_clip(clip_path(GENVIDEO), runner)
    out = runner.run(np.stack([frames, frames]), np.array([n_valid, n_valid]))
    np.testing.assert_allclose(out["fake_prob"][0], out["fake_prob"][1], atol=1e-6)  # identical clips, identical rows
    if model == "mars":
        assert np.all(out["frame_contrib"][0, n_valid:] == 0)
        assert np.all(out["patch_contrib"][0, n_valid:] == 0)
        np.testing.assert_allclose(out["gate"].sum(axis=1), 1.0, atol=1e-4)
    else:
        probs = out["frame_probs"][0]
        assert np.all(probs[n_valid:] == 0)
        p = np.clip(probs[:n_valid].astype(np.float64), 1e-7, 1 - 1e-7)
        expected = 1 / (1 + np.exp(-np.mean(np.log(p / (1 - p)))))
        assert float(out["fake_prob"][0]) == pytest.approx(expected, abs=2e-3)


def test_device_is_reported() -> None:
    runner = Runner(get_spec("gend"))
    assert runner.device in ("cuda", "cpu")
    if runner.device == "cpu":
        assert runner.precision == "fp32"
