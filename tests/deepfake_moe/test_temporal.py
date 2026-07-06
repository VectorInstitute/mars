"""Tests for temporal IoU metrics."""

from deepfake_moe.eval.metrics.temporal import compute_temporal_metrics, segment_iou


def test_segment_iou_identical() -> None:
    assert segment_iou([[1.0, 3.0]], [[1.0, 3.0]]) == 1.0


def test_segment_iou_partial_overlap() -> None:
    assert round(segment_iou([[0.0, 2.0]], [[1.0, 3.0]]), 4) == 0.3333


def test_segment_iou_both_empty() -> None:
    assert segment_iou([], []) == 1.0


def test_segment_iou_one_empty() -> None:
    assert segment_iou([[0.0, 1.0]], []) == 0.0
    assert segment_iou([], [[0.0, 1.0]]) == 0.0


def test_compute_temporal_metrics_skips_unparsed() -> None:
    records = [
        {
            "parse_ok": True,
            "pred_fake_periods": [[0.0, 2.0]],
            "fake_periods": [[1.0, 3.0]],
        },
        {"parse_ok": False, "pred_fake_periods": [], "fake_periods": []},
    ]
    metrics = compute_temporal_metrics(records)
    assert metrics["n_evaluated"] == 1
    assert metrics["mean_iou"] == round(segment_iou([[0.0, 2.0]], [[1.0, 3.0]]), 4)


def test_compute_temporal_metrics_both_empty_is_perfect() -> None:
    records = [{"parse_ok": True, "pred_fake_periods": [], "fake_periods": []}]
    metrics = compute_temporal_metrics(records)
    assert metrics["mean_iou"] == 1.0
    assert metrics["f1_at_iou_25"] == 1.0


def test_compute_temporal_metrics_empty_input() -> None:
    assert compute_temporal_metrics([])["n_evaluated"] == 0
