"""Turn per-clip predictions into the result payload every model reports."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from mars.eval.metrics import aggregate_to_videos, compute, compute_per_method, method_tables


def summarise(predictions: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Clip and video metrics, per-class tables, and the raw rows.

    predictions holds parallel lists of video_ids, clip_indices,
    labels, scores and methods. A video's score is the mean over its
    clips.
    """
    clip = compute(predictions["labels"], predictions["scores"])
    video_labels, video_scores = aggregate_to_videos(
        predictions["video_ids"], predictions["labels"], predictions["scores"]
    )
    video = compute(video_labels, video_scores)

    scores_by_video: dict[str, list[float]] = defaultdict(list)
    label_by_video: dict[str, int] = {}
    method_by_video: dict[str, str] = {}
    for video_id, score, label, method in zip(
        predictions["video_ids"], predictions["scores"], predictions["labels"], predictions["methods"]
    ):
        scores_by_video[video_id].append(score)
        label_by_video[video_id] = label
        method_by_video[video_id] = method

    order = sorted(scores_by_video)
    per_class: dict[str, Any] = {}
    if any(predictions["methods"]):
        per_class = method_tables(
            compute_per_method(predictions["labels"], predictions["scores"], predictions["methods"]),
            compute_per_method(
                [label_by_video[v] for v in order],
                [sum(scores_by_video[v]) / len(scores_by_video[v]) for v in order],
                [method_by_video[v] for v in order],
            ),
        )

    clips = [
        {
            "dataset": dataset,
            "video_id": video_id,
            "clip_index": clip_index,
            "label": label,
            "probability": score,
            **({"method": method} if method else {}),
        }
        for video_id, clip_index, label, score, method in zip(
            predictions["video_ids"],
            predictions["clip_indices"],
            predictions["labels"],
            predictions["scores"],
            predictions["methods"],
        )
    ]

    return {
        "dataset": dataset,
        "clip": clip,
        "video": video,
        "per_class": per_class,
        "clips": clips,
        "per_video": {
            video_id: {
                "label": label_by_video[video_id],
                "probability": sum(scores) / len(scores),
                "predicted": int(sum(scores) / len(scores) >= 0.5),
                **({"method": method_by_video[video_id]} if method_by_video[video_id] else {}),
                "clip_probabilities": scores,
            }
            for video_id, scores in sorted(scores_by_video.items())
        },
    }


def load_trainable_weights(model: torch.nn.Module, checkpoint: str, device: torch.device) -> None:
    """Load a checkpoint's trainable weights, allowing a frozen encoder to be absent."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys in checkpoint: {unexpected}")
    if any(not key.startswith("encoder.") for key in missing):
        raise RuntimeError(f"checkpoint is missing trainable weights: {missing}")
