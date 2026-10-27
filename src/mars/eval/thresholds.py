"""Operating points: the fixed 0.5 and a threshold chosen on source validation.

The source threshold maximises Youden's J (true positive rate minus false positive
rate) on the model's own source-validation videos, taking the lowest threshold
when several tie. It is chosen once and applied unchanged to every test corpus;
it is never fitted on test labels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from mars.eval.metrics import equal_error_rate


FIXED = 0.5
# J is tpr - fpr, so two genuinely tied thresholds can differ by one unit in the
# last place; an exact comparison would miss the tie and pick the higher one.
TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SourceThreshold:
    """A source-selected operating point and where it came from."""

    value: float
    n_videos: int
    n_real: int
    n_fake: int
    roc_auc: float
    youden_j: float
    n_maximisers: int

    def as_dict(self) -> dict[str, Any]:
        """Return the record saved beside the results."""
        return {
            "threshold": self.value,
            "rule": "lowest maximiser of Youden J on source validation videos",
            "n_videos": self.n_videos,
            "n_real": self.n_real,
            "n_fake": self.n_fake,
            "roc_auc": self.roc_auc,
            "youden_j": self.youden_j,
            "n_maximising_thresholds": self.n_maximisers,
        }


def load_source_threshold(path: str | Path) -> SourceThreshold:
    """Read a source threshold saved beside an earlier run's results."""
    saved = json.loads(Path(path).read_text())
    return SourceThreshold(
        value=float(saved["threshold"]),
        n_videos=int(saved["n_videos"]),
        n_real=int(saved["n_real"]),
        n_fake=int(saved["n_fake"]),
        roc_auc=float(saved["roc_auc"]),
        youden_j=float(saved["youden_j"]),
        n_maximisers=int(saved["n_maximising_thresholds"]),
    )


def source_threshold(labels: list[int], scores: list[float]) -> SourceThreshold:
    """Choose the operating point from validation labels and scores."""
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    fpr, tpr, thresholds = roc_curve(y, p)
    j = tpr - fpr
    best = np.flatnonzero(j >= j.max() - TIE_TOLERANCE)
    return SourceThreshold(
        value=float(thresholds[best].min()),
        n_videos=int(y.size),
        n_real=int((y == 0).sum()),
        n_fake=int((y == 1).sum()),
        roc_auc=float(roc_auc_score(y, p)),
        youden_j=float(j.max()),
        n_maximisers=int(best.size),
    )


def at_threshold(labels: list[int], scores: list[float], threshold: float) -> dict[str, Any]:
    """Accuracy, precision, recall, F1 and the confusion counts at one threshold."""
    y = np.asarray(labels, dtype=int)
    pred = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    n_fake, n_real = tp + fn, tn + fp
    nan = float("nan")
    precision = tp / (tp + fp) if tp + fp else nan
    recall = tp / n_fake if n_fake else nan
    specificity = tn / n_real if n_real else nan
    f1 = 2 * precision * recall / (precision + recall) if tp else (0.0 if n_fake else nan)
    return {
        "threshold": float(threshold),
        "accuracy": float((pred == y).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2 if n_fake and n_real else nan,
        "fpr": fp / n_real if n_real else nan,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "n": int(y.size),
        "n_fake": n_fake,
        "n_real": n_real,
    }


def ranking(labels: list[int], scores: list[float]) -> dict[str, Any]:
    """Threshold-free metrics; undefined when only one class is present.

    pr_auc is average precision, the step-wise sum, not a trapezoidal area.
    """
    y = np.asarray(labels, dtype=int)
    p = np.asarray(scores, dtype=float)
    if len(set(y.tolist())) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "eer": float("nan"), "n": int(y.size)}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "eer": equal_error_rate(y, p),
        "n": int(y.size),
    }


def report(labels: list[int], scores: list[float], source: SourceThreshold) -> dict[str, Any]:
    """Ranking once, then both operating points side by side."""
    return {
        "ranking": ranking(labels, scores),
        "at_0.5": at_threshold(labels, scores, FIXED),
        "at_source": at_threshold(labels, scores, source.value),
    }
