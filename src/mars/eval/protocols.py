"""Apply both operating points to evaluation results and write them out."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from mars.eval.metrics import macro
from mars.eval.thresholds import SourceThreshold, report


LEVELS = ("clip", "video")
PROTOCOLS = ("at_0.5", "at_source")
REPORTED = ("accuracy", "precision", "recall", "f1", "balanced_accuracy", "fpr", "tp", "fp", "tn", "fn")
AT_THRESHOLD = (("acc", "accuracy"), ("P", "precision"), ("R", "recall"), ("F1", "f1"))
PER_METHOD = ("roc_auc", "pr_auc", "eer", "recall_at_0.5", "n_fake", "n_real")


def labels_and_scores(result: dict[str, Any], level: str) -> tuple[list[int], list[float]]:
    """Labels and fake probabilities at clip or video level from a summarised result."""
    if level == "clip":
        rows = result["clips"]
        return [int(r["label"]) for r in rows], [float(r["probability"]) for r in rows]
    videos = result["per_video"].values()
    return [int(v["label"]) for v in videos], [float(v["probability"]) for v in videos]


def protocol_report(result: dict[str, Any], source: SourceThreshold) -> dict[str, Any]:
    """Both operating points, at clip and video level, for one corpus.

    A corpus with fake methods also gets each method scored against all reals, and
    the unweighted mean over methods.
    """
    out = {level: report(*labels_and_scores(result, level), source) for level in LEVELS}
    per_class = result.get("per_class") or {}
    if per_class:
        for level in LEVELS:
            out[level]["per_class"] = {
                name: {key: tables[level].get(key) for key in PER_METHOD} for name, tables in per_class.items()
            }
            out[level]["macro"] = macro(per_class, level)
    return out


def flat(prefix: str, protocols: dict[str, Any]) -> dict[str, float]:
    """One flat slash-joined mapping of every numeric leaf, for the logger."""
    out: dict[str, float] = {}
    for key, value in protocols.items():
        if isinstance(value, dict):
            out |= flat(f"{prefix}/{key}", value)
        elif isinstance(value, (int, float)):
            out[f"{prefix}/{key}"] = float(value)
    return out


def _cell(value: float | None) -> str:
    return "—" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def markdown(protocols: dict[str, dict[str, Any]], source: SourceThreshold) -> str:
    """Readable tables: one per level, one row per corpus, both operating points."""
    lines = [
        f"Source threshold **{source.value:.6f}** — Youden J {source.youden_j:.4f} on "
        f"{source.n_videos} source-validation videos ({source.n_real} real, {source.n_fake} fake), "
        f"ROC-AUC {source.roc_auc:.4f}.",
        "",
    ]
    for level in LEVELS:
        lines += [
            f"## {level.capitalize()} level",
            "",
            "| corpus | AUROC | AP | acc@0.5 | P@0.5 | R@0.5 | F1@0.5 | acc@src | P@src | R@src | F1@src "
            "| TP/FP/TN/FN @0.5 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for corpus, blocks in protocols.items():
            r, a, s = blocks[level]["ranking"], blocks[level]["at_0.5"], blocks[level]["at_source"]
            lines.append(
                f"| {corpus} | {_cell(r['roc_auc'])} | {_cell(r['pr_auc'])} | "
                f"{_cell(a['accuracy'])} | {_cell(a['precision'])} | {_cell(a['recall'])} | {_cell(a['f1'])} | "
                f"{_cell(s['accuracy'])} | {_cell(s['precision'])} | {_cell(s['recall'])} | {_cell(s['f1'])} | "
                f"{a['tp']}/{a['fp']}/{a['tn']}/{a['fn']} |"
            )
        lines.append("")
    for corpus, blocks in protocols.items():
        if "per_class" in blocks["video"]:
            lines += _per_class_table(corpus, blocks["video"])
    return "\n".join(lines)


def _counts(block: dict[str, Any]) -> str:
    return f"TP/FP/TN/FN {block['tp']}/{block['fp']}/{block['tn']}/{block['fn']}"


def _at(block: dict[str, Any]) -> str:
    return " ".join(f"{name} {_cell(block[key])}" for name, key in AT_THRESHOLD) + f"  {_counts(block)}"


def bullets(protocols: dict[str, dict[str, Any]], source: SourceThreshold) -> str:
    """Log-friendly summary: one block of bullets per corpus, video level first."""
    lines = [f"source threshold {source.value:.6f} (FF++ val AUROC {source.roc_auc:.4f}, {source.n_videos} videos)"]
    for corpus, blocks in protocols.items():
        video, clip = blocks["video"], blocks["clip"]
        r, cr = video["ranking"], clip["ranking"]
        lines += [
            "",
            f"TEST {corpus} ({r['n']} videos, {cr['n']} clips)",
            f"  - video ranking    AUROC {_cell(r['roc_auc'])}  AP {_cell(r['pr_auc'])}  EER {_cell(r['eer'])}",
            f"  - video at 0.5     {_at(video['at_0.5'])}",
            f"  - video at source  {_at(video['at_source'])}",
            f"  - clip ranking     AUROC {_cell(cr['roc_auc'])}  AP {_cell(cr['pr_auc'])}  EER {_cell(cr['eer'])}",
            f"  - clip at 0.5      {_at(clip['at_0.5'])}",
        ]
        if "per_class" in video:
            m = video["macro"]
            lines.append(
                f"  - per method, video, vs all reals (macro AUROC {_cell(m['roc_auc'])}  EER {_cell(m['eer'])})"
            )
            for name, row in video["per_class"].items():
                lines.append(
                    f"      - {name}: AUROC {_cell(row['roc_auc'])}  EER {_cell(row['eer'])}  "
                    f"R@0.5 {_cell(row['recall_at_0.5'])}  ({row['n_fake']} fake / {row['n_real']} real)"
                )
    return "\n".join(lines)


def _per_class_table(corpus: str, video: dict[str, Any]) -> list[str]:
    """Video-level table of each fake method against all reals, with the macro row."""
    lines = [
        f"## {corpus}: per method (video level, each vs all reals)",
        "",
        "| method | fake/real | AUROC | AP | EER | R@0.5 |",
        "|---|---|---|---|---|---|",
    ]
    for name, row in video["per_class"].items():
        lines.append(
            f"| {name} | {row['n_fake']}/{row['n_real']} | {_cell(row['roc_auc'])} | {_cell(row['pr_auc'])} | "
            f"{_cell(row['eer'])} | {_cell(row['recall_at_0.5'])} |"
        )
    m = video["macro"]
    lines += [f"| **macro** | | {_cell(m['roc_auc'])} | {_cell(m['pr_auc'])} | {_cell(m['eer'])} | |", ""]
    return lines


def write_protocols(
    results_dir: str | Path,
    protocols: dict[str, dict[str, Any]],
    source: SourceThreshold,
    source_validation: dict[str, Any] | None,
) -> None:
    """Write the threshold, every corpus under both operating points, and a summary.

    source_validation is None when the threshold was reused from an earlier run;
    no validation predictions are written then.
    """
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "source_threshold.json").write_text(json.dumps(source.as_dict(), indent=1))
    (root / "protocols.json").write_text(json.dumps(protocols, indent=1))
    (root / "summary.md").write_text(markdown(protocols, source))
    if source_validation is None:
        return
    with (root / "source_validation.jsonl").open("w") as handle:
        for video_id, row in sorted(source_validation["per_video"].items()):
            handle.write(json.dumps({"video_id": video_id, "label": row["label"], "probability": row["probability"]}))
            handle.write("\n")
