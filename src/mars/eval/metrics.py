"""Clip- and video-level detection metrics.

The positive class is fake. pr_auc is the area under the precision-recall
curve. roc_auc and eer are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


METRIC_ORDER = [
    "accuracy",
    "pr_auc",
    "precision_at_0.5",
    "recall_at_0.5",
    "precision_at_recall_0.8",
    "recall_at_precision_0.8",
    "roc_auc",
    "eer",
]


@dataclass
class Metrics:
    """Scalar detection metrics for one set of labels and scores."""

    accuracy: float
    pr_auc: float | None
    precision_at_05: float | None
    recall_at_05: float | None
    precision_at_recall_08: float | None
    recall_at_precision_08: float | None
    roc_auc: float | None
    eer: float | None
    n: int

    def as_dict(self, prefix: str = "") -> dict[str, float | int | None]:
        """Return metrics as a flat dict, optionally prefixed."""
        mapping = dict(
            zip(
                METRIC_ORDER,
                [
                    self.accuracy,
                    self.pr_auc,
                    self.precision_at_05,
                    self.recall_at_05,
                    self.precision_at_recall_08,
                    self.recall_at_precision_08,
                    self.roc_auc,
                    self.eer,
                ],
            )
        )
        mapping["n"] = self.n
        return {f"{prefix}{k}": v for k, v in mapping.items()}

    def table(self, title: str = "") -> str:
        """Format metrics as a short text table."""
        rows = [title] if title else []
        for key, value in self.as_dict().items():
            if key == "n":
                rows.append(f"  {key:<26}{value:>10d}")
            elif value is None:
                rows.append(f"  {key:<26}{'n/a':>10}")
            else:
                rows.append(f"  {key:<26}{value:>10.4f}")
        return "\n".join(rows)


@dataclass
class MethodMetrics:
    """One fake method scored against every real sample in the dataset."""

    accuracy: float
    precision_at_05: float | None
    recall_at_05: float | None
    roc_auc: float | None
    pr_auc: float | None
    eer: float | None
    n: int
    n_fake: int
    n_real: int

    def as_dict(self) -> dict[str, float | int | None]:
        """Return the per-method payload."""
        return {
            "accuracy": self.accuracy,
            "precision_at_0.5": self.precision_at_05,
            "recall_at_0.5": self.recall_at_05,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "eer": self.eer,
            "n": self.n,
            "n_fake": self.n_fake,
            "n_real": self.n_real,
        }


def _precision_at_recall(precision: np.ndarray, recall: np.ndarray, target: float) -> float:
    feasible = recall >= target
    return float(precision[feasible].max()) if feasible.any() else 0.0


def _recall_at_precision(precision: np.ndarray, recall: np.ndarray, target: float) -> float:
    feasible = precision >= target
    return float(recall[feasible].max()) if feasible.any() else 0.0


def _threshold_precision_recall(
    labels_arr: np.ndarray,
    predictions: np.ndarray,
    positive: int,
) -> tuple[float | None, float | None]:
    if positive == 0:
        return None, None
    true_positive = int(((predictions == 1) & (labels_arr == 1)).sum())
    predicted_positive = int((predictions == 1).sum())
    precision_05 = true_positive / predicted_positive if predicted_positive else 0.0
    recall_05 = true_positive / int(positive)
    return precision_05, recall_05


def equal_error_rate(labels: object, scores: object) -> float:
    """Error rate where the false-positive and false-negative rates meet."""
    fpr, tpr, _ = roc_curve(np.asarray(labels).astype(int).ravel(), np.asarray(scores, dtype=float).ravel())
    fnr = 1.0 - tpr
    index = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[index] + fnr[index]) / 2.0)


def compute(labels: object, scores: object, threshold: float = 0.5) -> Metrics:
    """Score binary predictions. labels are 0/1 (1 = fake); scores are P(fake)."""
    labels_arr = np.asarray(labels).astype(int).ravel()
    scores_arr = np.asarray(scores, dtype=float).ravel()
    if labels_arr.size == 0:
        raise ValueError("no samples to score")

    predictions = (scores_arr >= threshold).astype(int)
    accuracy = float((predictions == labels_arr).mean())
    positive = int(labels_arr.sum())
    precision_05, recall_05 = _threshold_precision_recall(labels_arr, predictions, positive)
    n = int(labels_arr.size)

    if positive in (0, n):
        return Metrics(accuracy, None, precision_05, recall_05, None, None, None, None, n)

    precision, recall, _ = precision_recall_curve(labels_arr, scores_arr)
    return Metrics(
        accuracy=accuracy,
        pr_auc=float(average_precision_score(labels_arr, scores_arr)),
        precision_at_05=precision_05,
        recall_at_05=recall_05,
        precision_at_recall_08=_precision_at_recall(precision, recall, 0.8),
        recall_at_precision_08=_recall_at_precision(precision, recall, 0.8),
        roc_auc=float(roc_auc_score(labels_arr, scores_arr)),
        eer=equal_error_rate(labels_arr, scores_arr),
        n=n,
    )


def compute_per_method(
    labels: object,
    scores: object,
    methods: object,
    threshold: float = 0.5,
) -> dict[str, MethodMetrics]:
    """Score each fake method together with the real samples in the same arrays."""
    labels_arr = np.asarray(labels).astype(int).ravel()
    scores_arr = np.asarray(scores, dtype=float).ravel()
    methods_arr = np.asarray(methods).astype(str).ravel()
    real_mask = labels_arr == 0
    n_real = int(real_mask.sum())
    out: dict[str, MethodMetrics] = {}
    fake_methods = sorted(
        {name for name, label in zip(methods_arr.tolist(), labels_arr.tolist()) if label == 1 and name}
    )
    for name in fake_methods:
        method_mask = methods_arr == name
        n_fake = int((method_mask & (labels_arr == 1)).sum())
        pool = method_mask | real_mask
        metrics = compute(labels_arr[pool], scores_arr[pool], threshold=threshold)
        out[name] = MethodMetrics(
            accuracy=metrics.accuracy,
            precision_at_05=metrics.precision_at_05,
            recall_at_05=metrics.recall_at_05,
            roc_auc=metrics.roc_auc,
            pr_auc=metrics.pr_auc,
            eer=metrics.eer,
            n=int(pool.sum()),
            n_fake=n_fake,
            n_real=n_real,
        )
    return out


def method_tables(
    clip_by: dict[str, MethodMetrics],
    video_by: dict[str, MethodMetrics],
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """Join clip and video per-method tables."""
    names = sorted(set(clip_by) | set(video_by))
    return {
        name: {
            "clip": clip_by[name].as_dict() if name in clip_by else {},
            "video": video_by[name].as_dict() if name in video_by else {},
        }
        for name in names
    }


def macro(per_class: dict[str, dict[str, dict[str, float | int | None]]], level: str) -> dict[str, float | None]:
    """Unweighted mean of each ranking metric over the methods of a method table."""
    out: dict[str, float | None] = {}
    for key in ("roc_auc", "pr_auc", "eer"):
        values = [float(v) for table in per_class.values() if (v := table.get(level, {}).get(key)) is not None]
        out[key] = sum(values) / len(values) if values else None
    return out


def aggregate_to_videos(
    video_ids: object,
    labels: object,
    scores: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of clip probabilities per video."""
    video_ids_arr = np.asarray(video_ids)
    labels_arr = np.asarray(labels).astype(int)
    scores_arr = np.asarray(scores, dtype=float)

    unique = np.unique(video_ids_arr)
    video_labels = np.empty(unique.size, dtype=int)
    video_scores = np.empty(unique.size, dtype=float)
    for i, video in enumerate(unique):
        mask = video_ids_arr == video
        video_labels[i] = labels_arr[mask][0]
        video_scores[i] = scores_arr[mask].mean()
    return video_labels, video_scores
