"""Temporal segment IoU metrics."""

from __future__ import annotations

import numpy as np


def segment_iou(pred: list[list[float]], gt: list[list[float]]) -> float:
    """Temporal IoU between two sets of time segments.

    Parameters
    ----------
    pred :
        Predicted segments ``[[t0, t1], ...]`` in seconds.
    gt :
        Ground-truth segments ``[[t0, t1], ...]`` in seconds.

    Returns
    -------
    float
        IoU in ``[0.0, 1.0]``; both empty → ``1.0``, one empty → ``0.0``.

    Examples
    --------
    >>> segment_iou([[1.0, 3.0]], [[1.0, 3.0]])
    1.0
    >>> segment_iou([], [])
    1.0
    >>> segment_iou([[1.0, 2.0]], [])
    0.0
    >>> round(segment_iou([[0.0, 2.0]], [[1.0, 3.0]]), 4)
    0.3333
    """
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0

    inter = _total_overlap(pred, gt)
    union = _total_length(pred) + _total_length(gt) - inter
    return inter / union if union > 0.0 else 0.0


def compute_temporal_metrics(records: list[dict]) -> dict:
    """Temporal localisation metrics for a set of clips.

    Only ``parse_ok=True`` records are included. Clips where both
    ``pred_fake_periods`` and ``fake_periods`` are empty count as IoU = 1.0.

    Returns
    -------
    dict
        ``n_evaluated``, ``mean_iou``, ``f1_at_iou_25``, ``f1_at_iou_50``.
    """
    valid = [r for r in records if r.get("parse_ok", False)]

    if not valid:
        return {"n_evaluated": 0, "note": "no valid records"}

    ious = np.array(
        [segment_iou(r.get("pred_fake_periods") or [], r.get("fake_periods") or []) for r in valid],
        dtype=float,
    )

    return {
        "n_evaluated": len(valid),
        "mean_iou": round(float(ious.mean()), 4),
        "f1_at_iou_25": round(float((ious >= 0.25).mean()), 4),
        "f1_at_iou_50": round(float((ious >= 0.50).mean()), 4),
    }


def _total_overlap(segs_a: list[list[float]], segs_b: list[list[float]]) -> float:
    total = 0.0
    for a0, a1 in segs_a:
        for b0, b1 in segs_b:
            total += max(0.0, min(a1, b1) - max(a0, b0))
    return total


def _total_length(segs: list[list[float]]) -> float:
    return sum(t1 - t0 for t0, t1 in segs)
