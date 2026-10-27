"""Result writer layout and metric pooling."""

from __future__ import annotations

import json
from pathlib import Path

from mars.eval.metrics import compute, compute_per_method, equal_error_rate, macro, method_tables
from mars.eval.write import recompute_experiment_results, write_experiment_results


def test_compute_mixed_pool_has_ranking_metrics() -> None:
    metrics = compute([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    assert metrics.n == 4
    assert metrics.pr_auc is not None and metrics.pr_auc > 0.9
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.9
    assert metrics.precision_at_05 == 1.0
    assert metrics.recall_at_05 == 1.0


def test_compute_single_class_omits_ranking_metrics() -> None:
    fakes = compute([1, 1, 1], [0.9, 0.8, 0.7])
    assert fakes.accuracy == 1.0
    assert fakes.pr_auc is None
    assert fakes.roc_auc is None
    assert fakes.precision_at_recall_08 is None
    assert fakes.recall_at_precision_08 is None
    assert fakes.recall_at_05 == 1.0

    reals = compute([0, 0, 0], [0.1, 0.2, 0.3])
    assert reals.accuracy == 1.0
    assert reals.pr_auc is None
    assert reals.recall_at_05 is None


def test_compute_per_method_pairs_fakes_with_reals() -> None:
    labels = [0, 0, 1, 1, 1]
    scores = [0.1, 0.6, 0.9, 0.8, 0.4]
    methods = ["youtube", "youtube", "Deepfakes", "Deepfakes", "Face2Face"]
    by_method = compute_per_method(labels, scores, methods)
    assert set(by_method) == {"Deepfakes", "Face2Face"}
    deep = by_method["Deepfakes"]
    assert deep.n_fake == 2
    assert deep.n_real == 2
    assert deep.n == 4
    assert deep.precision_at_05 == 2 / 3
    assert deep.recall_at_05 == 1.0
    face = by_method["Face2Face"]
    assert face.n_fake == 1
    assert face.recall_at_05 == 0.0
    assert deep.roc_auc == 1.0
    assert deep.eer == 0.0
    assert face.roc_auc == 0.5


def test_equal_error_rate_matches_hand_count() -> None:
    # At threshold 0.7 one of four reals passes and one of four fakes fails.
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.7, 0.4, 0.8, 0.9, 0.95]
    assert equal_error_rate(labels, scores) == 0.25
    assert compute(labels, scores).eer == 0.25


def test_macro_is_the_unweighted_mean_over_methods() -> None:
    labels = [0, 0, 1, 1, 1]
    scores = [0.1, 0.6, 0.9, 0.8, 0.4]
    methods = ["real", "real", "A", "A", "B"]
    by_method = compute_per_method(labels, scores, methods)
    tables = method_tables(by_method, by_method)
    assert macro(tables, "video")["roc_auc"] == (1.0 + 0.5) / 2


def test_results_writer_dataset_and_overall(tmp_path: Path) -> None:
    clip = compute([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2])
    video = compute([0, 1], [0.15, 0.85])
    per_class = {
        name: {
            "clip": metrics.as_dict(),
            "video": metrics.as_dict(),
        }
        for name, metrics in compute_per_method(
            [0, 1],
            [0.15, 0.85],
            ["youtube", "Deepfakes"],
        ).items()
    }
    result = {
        "dataset": "ffpp",
        "clip": clip,
        "video": video,
        "per_class": per_class,
        "clips": [
            {"dataset": "ffpp", "video_id": "a", "clip_index": 0, "label": 0, "probability": 0.1, "method": "youtube"},
            {
                "dataset": "ffpp",
                "video_id": "b",
                "clip_index": 0,
                "label": 1,
                "probability": 0.9,
                "method": "Deepfakes",
            },
        ],
        "per_video": {
            "a": {"label": 0, "probability": 0.1, "predicted": 0, "method": "youtube", "clip_probabilities": [0.1]},
            "b": {"label": 1, "probability": 0.9, "predicted": 1, "method": "Deepfakes", "clip_probabilities": [0.9]},
        },
    }
    root = write_experiment_results(results_dir=tmp_path / "unite" / "exp", dataset_results={"ffpp": result})
    assert (root / "ffpp" / "predictions.jsonl").exists()
    dataset_metrics = json.loads((root / "ffpp" / "metrics.json").read_text())
    assert "Deepfakes" in dataset_metrics["per_class"]
    assert "youtube" not in dataset_metrics["per_class"]
    assert dataset_metrics["per_class"]["Deepfakes"]["video"]["n_real"] == 1
    overall = json.loads((root / "overall" / "metrics.json").read_text())
    assert "per_class" not in overall
    assert overall["clip"]["pr_auc"] is not None
    assert (root / "overall" / "summary.json").exists()


def test_recompute_from_saved_predictions(tmp_path: Path) -> None:
    clip = compute([0, 1], [0.1, 0.9])
    video = compute([0, 1], [0.1, 0.9])
    result = {
        "dataset": "ffpp",
        "clip": clip,
        "video": video,
        "per_class": {},
        "clips": [
            {"dataset": "ffpp", "video_id": "a", "clip_index": 0, "label": 0, "probability": 0.1, "method": "youtube"},
            {
                "dataset": "ffpp",
                "video_id": "b",
                "clip_index": 0,
                "label": 1,
                "probability": 0.9,
                "method": "Deepfakes",
            },
        ],
        "per_video": {
            "a": {"label": 0, "probability": 0.1, "predicted": 0, "method": "youtube", "clip_probabilities": [0.1]},
            "b": {"label": 1, "probability": 0.9, "predicted": 1, "method": "Deepfakes", "clip_probabilities": [0.9]},
        },
    }
    root = write_experiment_results(results_dir=tmp_path / "exp", dataset_results={"ffpp": result})
    (root / "ffpp" / "metrics.json").write_text("{}\n")
    recompute_experiment_results(root)
    metrics = json.loads((root / "ffpp" / "metrics.json").read_text())
    assert metrics["per_class"]["Deepfakes"]["video"]["recall_at_0.5"] == 1.0
    assert "per_class" not in json.loads((root / "overall" / "metrics.json").read_text())


def test_source_threshold_round_trips_through_its_file(tmp_path: Path) -> None:
    from mars.eval.protocols import write_protocols
    from mars.eval.thresholds import SourceThreshold, load_source_threshold

    source = SourceThreshold(0.37, 700, 140, 560, 0.99, 0.92, 1)
    write_protocols(tmp_path, {}, source, None)
    assert load_source_threshold(tmp_path / "source_threshold.json") == source
    # a reused threshold writes no validation predictions
    assert not (tmp_path / "source_validation.jsonl").exists()
