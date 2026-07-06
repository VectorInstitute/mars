"""Aggregate metrics, Phase 1 gates, and report files."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from deepfake_moe.eval.metrics.detection import compute_detection_metrics
from deepfake_moe.eval.metrics.temporal import compute_temporal_metrics
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

_PARSE_RATE_GATE = 0.80
_AUC_GATE = 0.50


def generate_report(
    records: list[dict],
    temporal_datasets: list[str],
    output_dir: Path,
) -> dict:
    """Compute metrics, check Phase 1 gates, write ``metrics.json`` and ``report.md``.

    Parameters
    ----------
    records :
        Per-clip results from ``run_eval``.
    temporal_datasets :
        Dataset names where temporal IoU is computed.
    output_dir :
        Directory for output files (created if absent).

    Returns
    -------
    dict
        Full metrics summary (same content as ``metrics.json``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_dataset[r["dataset"]].append(r)

    dataset_metrics: dict[str, dict] = {}
    for dataset in sorted(by_dataset):
        ds_records = by_dataset[dataset]
        parse_rate = _parse_rate(ds_records)
        entry: dict = {
            "n_clips": len(ds_records),
            "parse_rate": round(parse_rate, 4),
            "video": compute_detection_metrics(ds_records, "video"),
            "audio": compute_detection_metrics(ds_records, "audio"),
        }
        if dataset in temporal_datasets:
            entry["temporal"] = compute_temporal_metrics(ds_records)
        dataset_metrics[dataset] = entry

    summary: dict = {
        "n_total": len(records),
        "parse_rate": round(_parse_rate(records), 4),
        "video": compute_detection_metrics(records, "video"),
        "audio": compute_detection_metrics(records, "audio"),
        "by_dataset": dataset_metrics,
    }

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info("Metrics → %s", metrics_path)

    md = _build_markdown(summary, temporal_datasets)
    report_path = output_dir / "report.md"
    report_path.write_text(md)
    logger.info("Report  → %s", report_path)

    print(md)
    _check_gates(summary)

    return summary


def _parse_rate(records: list[dict]) -> float:
    if not records:
        return 0.0
    return sum(r.get("parse_ok", False) for r in records) / len(records)


def _fmt(value: float | None, decimals: int = 4) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _build_markdown(summary: dict, temporal_datasets: list[str]) -> str:
    lines: list[str] = ["# Phase 1 eval report\n"]

    pr = summary["parse_rate"]
    vm = summary["video"]
    am = summary["audio"]
    lines += [
        "## Overall\n",
        f"- Clips evaluated : {summary['n_total']}",
        f"- Parse rate      : {pr:.1%}",
        f"- Video — AUC {_fmt(vm.get('auc'))}  Acc {_fmt(vm.get('accuracy'))}  F1 {_fmt(vm.get('f1'))}",
        f"- Audio — AUC {_fmt(am.get('auc'))}  Acc {_fmt(am.get('accuracy'))}  F1 {_fmt(am.get('f1'))}",
        "",
    ]

    lines.append("## Per-dataset\n")
    header = "| Dataset | N | Parse% | Vid AUC | Vid Acc | Vid F1 | Aud AUC | Aud Acc | Aud F1 | mIoU |"
    sep = "|" + "|".join(["-" * len(c) for c in header.split("|")[1:-1]]) + "|"
    lines += [header, sep]

    for dataset, dm in summary["by_dataset"].items():
        v = dm["video"]
        a = dm["audio"]
        t = dm.get("temporal", {})
        row = [
            dataset,
            str(dm["n_clips"]),
            f"{dm['parse_rate']:.0%}",
            _fmt(v.get("auc")),
            _fmt(v.get("accuracy")),
            _fmt(v.get("f1")),
            _fmt(a.get("auc")),
            _fmt(a.get("accuracy")),
            _fmt(a.get("f1")),
            _fmt(t.get("mean_iou")) if t else "—",
        ]
        lines.append("| " + " | ".join(row) + " |")

    pr_ok = pr >= _PARSE_RATE_GATE
    v_auc = vm.get("auc")
    a_auc = am.get("auc")
    v_ok = v_auc is not None and v_auc > _AUC_GATE
    a_ok = a_auc is not None and a_auc > _AUC_GATE
    all_ok = pr_ok and v_ok and a_ok

    lines += [
        "",
        "## Phase 1 gates\n",
        "| Gate | Result | Value |",
        "|------|--------|-------|",
        f"| Parse rate ≥ 80% | {'✅ pass' if pr_ok else '❌ fail'} | {pr:.1%} |",
        f"| Video AUC > 0.50  | {'✅ pass' if v_ok else '❌ fail'} | {v_auc} |",
        f"| Audio AUC > 0.50  | {'✅ pass' if a_ok else '❌ fail'} | {a_auc} |",
        "",
        "**Decision:** "
        + (
            "All gates pass — proceed to MoE training with Qwen3-Omni backbone. ✅"
            if all_ok
            else "One or more gates failed — review perception stack before training. ❌"
        ),
        "",
    ]

    return "\n".join(lines)


def _check_gates(summary: dict) -> None:
    pr = summary["parse_rate"]
    v_auc = summary["video"].get("auc")
    a_auc = summary["audio"].get("auc")

    if pr < _PARSE_RATE_GATE:
        logger.warning(
            "GATE FAIL: parse rate %.1f%% < 80%%. Model is not reliably following structured output format.",
            100 * pr,
        )
    if v_auc is None or v_auc <= _AUC_GATE:
        logger.warning(
            "GATE FAIL: video AUC %s ≤ 0.50. Video head is no better than random.",
            v_auc,
        )
    if a_auc is None or a_auc <= _AUC_GATE:
        logger.warning(
            "GATE FAIL: audio AUC %s ≤ 0.50. Audio head is no better than random.",
            a_auc,
        )
