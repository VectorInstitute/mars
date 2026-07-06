"""Tests for eval runner with a mock predictor."""

import json
from pathlib import Path

import numpy as np
import pytest

from deepfake_moe.eval.dataset import EvalConfig
from deepfake_moe.eval.runner import run_eval
from deepfake_moe.models.base import ClipData, PredictionOutput
from deepfake_moe.models.registry import load_config


class _FakePredictor:
    def predict(self, clip: ClipData) -> PredictionOutput:
        return PredictionOutput(
            video_fake=clip.label_video == 1,
            audio_fake=False,
            video_confidence=7,
            audio_confidence=0,
            fake_periods=[],
            reason="mock",
            raw_response="{}",
            parse_ok=True,
        )


def test_run_eval_writes_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = {
        "clip_id": "clip-1",
        "dataset": "tiny",
        "src_path": "",
        "wav_path": None,
        "has_audio": False,
        "label": [0, 0],
        "fake_periods": [],
    }

    clip = ClipData(
        clip_id="clip-1",
        dataset="tiny",
        has_audio=False,
        frames=np.zeros((2, 4, 4, 3), dtype=np.uint8),
        waveform=None,
        label_video=0,
        label_audio=0,
        fake_periods=[],
    )

    class _LoaderPredictor(_FakePredictor):
        def predict(self, clip: ClipData) -> PredictionOutput:
            assert clip.clip_id == "clip-1"
            return super().predict(clip)

    def _fake_load_clip(entry, *, num_frames, sample_rate):
        return clip if entry["clip_id"] == "clip-1" else None

    monkeypatch.setattr("deepfake_moe.eval.runner.load_clip", _fake_load_clip)

    eval_cfg = EvalConfig(
        manifest_dir=str(tmp_path),
        dataset_sources={},
        num_clips_per_dataset=1,
        num_frames=2,
        seed=0,
        output_dir=str(tmp_path / "out"),
        temporal_datasets=[],
        model="qwen3-omni",
    )
    model_cfg = load_config("qwen3-omni-hf")
    out_path = tmp_path / "preds.jsonl"

    records = run_eval(_LoaderPredictor(), [entry], eval_cfg, model_cfg, out_path)
    assert len(records) == 1
    assert records[0]["parse_ok"]
    assert records[0]["video_fake"] is False

    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["clip_id"] == "clip-1"
