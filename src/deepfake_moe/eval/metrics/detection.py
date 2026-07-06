"""Per-head binary classification metrics (video / audio)."""

from __future__ import annotations

import numpy as np


try:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    _SKLEARN = True
except ImportError:
    _SKLEARN = False


def compute_detection_metrics(records: list[dict], head: str) -> dict:
    """Binary classification metrics for one detection head.

    Parameters
    ----------
    records :
        Per-clip result dicts from the JSONL output.
    head :
        ``"video"`` or ``"audio"``.

    Returns
    -------
    dict
        Keys include ``n_total``, ``n_evaluated``, ``accuracy``, ``f1``,
        ``precision``, ``recall``, and ``auc`` (``None`` if sklearn is absent
        or only one class is present).
    """
    if head not in ("video", "audio"):
        raise ValueError(f"head must be 'video' or 'audio', got {head!r}")

    n_total = len(records)
    valid = [r for r in records if r.get("parse_ok", False)]

    if head == "audio":
        valid = [r for r in valid if r.get("has_audio", False)]

    result: dict = {"n_total": n_total, "n_evaluated": len(valid)}

    if not valid:
        result["note"] = "no valid records for this head"
        return result

    labels = np.array([r[f"label_{head}"] for r in valid], dtype=int)
    preds = np.array([int(r[f"{head}_fake"]) for r in valid], dtype=int)
    confs = np.array([r[f"{head}_confidence"] for r in valid], dtype=float)
    scores = np.where(preds == 1, confs / 10.0, 1.0 - confs / 10.0)

    if _SKLEARN:
        result["accuracy"] = round(float(accuracy_score(labels, preds)), 4)
        result["f1"] = round(float(f1_score(labels, preds, zero_division=0)), 4)
        result["precision"] = round(float(precision_score(labels, preds, zero_division=0)), 4)
        result["recall"] = round(float(recall_score(labels, preds, zero_division=0)), 4)

        if len(np.unique(labels)) < 2:
            result["auc"] = None
            result["auc_note"] = "only one class present in labels"
        else:
            result["auc"] = round(float(roc_auc_score(labels, scores)), 4)
    else:
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        result["accuracy"] = round((tp + tn) / len(valid), 4)
        result["precision"] = round(prec, 4)
        result["recall"] = round(rec, 4)
        result["f1"] = round(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0, 4)
        result["auc"] = None
        result["auc_note"] = "sklearn not installed"

    return result
