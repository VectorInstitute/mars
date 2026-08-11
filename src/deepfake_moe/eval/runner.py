"""Eval loop: load clips, run ``Predictor``, write JSONL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TextIO

from deepfake_moe.eval.dataset import EvalConfig, load_clip
from deepfake_moe.models.base import ClipData, PredictionOutput, Predictor
from deepfake_moe.models.config import Config
from deepfake_moe.models.vllm_config import VllmConfig
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)


def run_eval(
    model: Predictor,
    entries: list[dict],
    eval_cfg: EvalConfig,
    model_cfg: Config | VllmConfig,
    output_path: Path,
) -> list[dict]:
    """Run the eval loop and write results to ``output_path`` as JSONL.

    Parameters
    ----------
    model :
        Any object implementing the ``Predictor`` protocol.
    entries :
        Manifest dicts from ``load_manifest_entries``.
    eval_cfg :
        Eval settings (frames, output paths, etc.).
    model_cfg :
        Model config supplying ``audio_sample_rate``.
    output_path :
        Destination JSONL; parent dirs are created. Each line is flushed immediately.

    Returns
    -------
    list[dict]
        All result records (same content as the JSONL file).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    n = len(entries)

    with open(output_path, "w") as fh:
        for i, entry in enumerate(entries):
            clip_id = entry.get("clip_id", f"clip_{i}")

            clip = load_clip(
                entry,
                num_frames=eval_cfg.num_frames,
                sample_rate=model_cfg.audio_sample_rate,
                data_root=eval_cfg.data_root,
                audio_dir=eval_cfg.resolved_audio_dir(),
                dataset_video_roots=eval_cfg.dataset_video_roots,
            )
            if clip is None:
                record = _error_record(entry, reason="load_failed")
                _write(fh, record)
                results.append(record)
                continue

            t0 = time.monotonic()
            try:
                pred = model.predict(clip)
            except Exception as exc:
                logger.error("predict failed  clip=%s  error=%s", clip_id, exc)
                record = _error_record(entry, reason=f"predict_failed: {exc}")
                _write(fh, record)
                results.append(record)
                continue

            elapsed = time.monotonic() - t0
            record = _make_record(clip, pred, elapsed)
            _write(fh, record)
            results.append(record)

            if (i + 1) % 50 == 0:
                n_ok = sum(r.get("parse_ok", False) for r in results)
                logger.info(
                    "[%d/%d]  dataset=%-12s  parse_ok=%.0f%%  last=%.1fs",
                    i + 1,
                    n,
                    entry.get("dataset", "?"),
                    100 * n_ok / len(results),
                    elapsed,
                )

    n_ok = sum(r.get("parse_ok", False) for r in results)
    logger.info(
        "Eval complete — total=%d  parse_ok=%d (%.0f%%)",
        n,
        n_ok,
        100 * n_ok / max(1, n),
    )
    return results


def _make_record(clip: ClipData, pred: PredictionOutput, elapsed: float) -> dict:
    return {
        "clip_id": clip.clip_id,
        "dataset": clip.dataset,
        "label_video": clip.label_video,
        "label_audio": clip.label_audio,
        "has_audio": clip.has_audio,
        "fake_periods": clip.fake_periods,
        "pred_fake_periods": pred.fake_periods,
        "video_fake": pred.video_fake,
        "audio_fake": pred.audio_fake,
        "video_fake_prob": pred.video_fake_prob,
        "audio_fake_prob": pred.audio_fake_prob,
        "evidence": pred.evidence,
        "reason": pred.reason,
        "parse_ok": pred.parse_ok,
        "elapsed_s": round(elapsed, 3),
    }


def _error_record(entry: dict, reason: str) -> dict:
    label = entry.get("label") or [0, 0]
    return {
        "clip_id": entry.get("clip_id", ""),
        "dataset": entry.get("dataset", ""),
        "label_video": int(label[0]),
        "label_audio": int(label[1]),
        "has_audio": bool(entry.get("has_audio", False)),
        "fake_periods": entry.get("fake_periods") or [],
        "video_fake": False,
        "audio_fake": False,
        "video_fake_prob": 0.0,
        "audio_fake_prob": 0.0,
        "evidence": [],
        "pred_fake_periods": [],
        "reason": reason,
        "parse_ok": False,
        "elapsed_s": 0.0,
    }


def _write(fh: TextIO, record: dict) -> None:
    fh.write(json.dumps(record) + "\n")
    fh.flush()
