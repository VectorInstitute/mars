"""Operating points and the two-protocol report."""

from __future__ import annotations

import math

from mars.eval.protocols import bullets, flat, markdown, protocol_report
from mars.eval.thresholds import at_threshold, ranking, source_threshold


def test_source_threshold_separates_a_clean_split() -> None:
    source = source_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert source.youden_j == 1.0
    assert 0.2 < source.value <= 0.8
    assert (source.n_real, source.n_fake) == (2, 2)


def test_ties_take_the_lowest_maximising_threshold() -> None:
    labels = [0, 1, 0, 1]
    scores = [0.2, 0.4, 0.6, 0.8]
    source = source_threshold(labels, scores)
    assert source.n_maximisers >= 2
    assert source.value == min(t for t in (0.4, 0.8) if t >= source.value)


def test_confusion_counts_and_rates() -> None:
    out = at_threshold([0, 0, 1, 1, 1], [0.1, 0.7, 0.9, 0.6, 0.2], 0.5)
    assert (out["tp"], out["fp"], out["tn"], out["fn"]) == (2, 1, 1, 1)
    assert math.isclose(out["precision"], 2 / 3)
    assert math.isclose(out["recall"], 2 / 3)
    assert math.isclose(out["fpr"], 1 / 2)
    assert math.isclose(out["accuracy"], 3 / 5)


def test_all_fake_corpus_has_recall_but_no_ranking() -> None:
    out = at_threshold([1, 1, 1], [0.9, 0.2, 0.7], 0.5)
    assert math.isclose(out["recall"], 2 / 3)
    assert math.isnan(out["fpr"])
    assert math.isnan(ranking([1, 1, 1], [0.9, 0.2, 0.7])["roc_auc"])


def test_no_positive_predictions_give_zero_f1_not_an_error() -> None:
    out = at_threshold([0, 1], [0.1, 0.2], 0.5)
    assert out["f1"] == 0.0
    assert math.isnan(out["precision"])


def _result() -> dict:
    clips = [
        {"label": 0, "probability": 0.1},
        {"label": 0, "probability": 0.6},
        {"label": 1, "probability": 0.8},
        {"label": 1, "probability": 0.4},
    ]
    videos = {"a": {"label": 0, "probability": 0.35}, "b": {"label": 1, "probability": 0.6}}
    return {"clips": clips, "per_video": videos}


def test_protocol_report_covers_both_levels_and_both_thresholds() -> None:
    source = source_threshold([0, 1], [0.35, 0.6])
    report = protocol_report(_result(), source)
    assert set(report) == {"clip", "video"}
    for level in report.values():
        assert set(level) == {"ranking", "at_0.5", "at_source"}
    assert report["clip"]["at_0.5"]["n"] == 4
    assert report["video"]["at_0.5"]["n"] == 2


def test_flat_keys_are_logger_ready() -> None:
    source = source_threshold([0, 1], [0.35, 0.6])
    keys = flat("test/ffpp", protocol_report(_result(), source))
    assert "test/ffpp/video/at_source/recall" in keys
    assert "test/ffpp/clip/ranking/roc_auc" in keys
    assert all(isinstance(v, float) for v in keys.values())


def test_methods_add_per_class_and_macro_blocks() -> None:
    source = source_threshold([0, 1], [0.35, 0.6])
    table = {"roc_auc": 0.75, "pr_auc": 0.7, "eer": 0.25, "recall_at_0.5": 0.5, "n_fake": 1, "n_real": 1}
    result = _result() | {"per_class": {"Veo": {"clip": table, "video": table}}}
    report = protocol_report(result, source)
    assert report["video"]["per_class"]["Veo"]["eer"] == 0.25
    assert report["video"]["macro"]["roc_auc"] == 0.75
    keys = flat("test/df26", report)
    assert keys["test/df26/video/per_class/Veo/roc_auc"] == 0.75
    assert keys["test/df26/video/macro/eer"] == 0.25
    assert "| **macro** |" in markdown({"df26": report}, source)


def test_bullets_give_one_block_per_corpus() -> None:
    source = source_threshold([0, 1], [0.35, 0.6])
    text = bullets({"ffpp": protocol_report(_result(), source)}, source)
    assert text.count("TEST ffpp") == 1
    assert "video at source" in text
    assert "|" not in text


def test_markdown_has_one_row_per_corpus_per_level() -> None:
    source = source_threshold([0, 1], [0.35, 0.6])
    text = markdown({"ffpp": protocol_report(_result(), source)}, source)
    assert text.count("| ffpp |") == 2
    assert "Source threshold" in text
