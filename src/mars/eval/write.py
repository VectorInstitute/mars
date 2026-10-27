"""Write per-dataset result folders plus a micro-averaged overall folder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mars.eval.metrics import Metrics, compute, compute_per_method, macro, method_tables


def _metrics_payload(metrics: Metrics) -> dict[str, float | int | None]:
    return metrics.as_dict()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dataset_metrics(result: dict[str, Any]) -> dict[str, Any]:
    clip: Metrics = result["clip"]
    video: Metrics = result["video"]
    per_class = result.get("per_class", {})
    payload = {
        "dataset": result["dataset"],
        "n_videos": video.n,
        "n_clips": clip.n,
        "clip": _metrics_payload(clip),
        "video": _metrics_payload(video),
        "per_class": per_class,
    }
    if per_class:
        payload["per_class_macro"] = {level: macro(per_class, level) for level in ("clip", "video")}
    return payload


def _per_class_from_rows(clip_rows: list[dict[str, Any]], video_rows: list[dict[str, Any]]) -> dict[str, Any]:
    clip_methods = [str(row.get("method") or "") for row in clip_rows]
    video_methods = [str(row.get("method") or "") for row in video_rows]
    if not any(clip_methods) and not any(video_methods):
        return {}
    return method_tables(
        compute_per_method(
            [int(row["label"]) for row in clip_rows],
            [float(row["probability"]) for row in clip_rows],
            clip_methods,
        )
        if clip_rows
        else {},
        compute_per_method(
            [int(row["label"]) for row in video_rows],
            [float(row["probability"]) for row in video_rows],
            video_methods,
        )
        if video_rows
        else {},
    )


def result_from_prediction_rows(
    dataset: str,
    clip_rows: list[dict[str, Any]],
    video_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rebuild an evaluate() payload from saved prediction files."""
    if not video_rows:
        raise ValueError(f"no video predictions for {dataset!r}")
    if not clip_rows:
        clip_rows = [
            {
                "dataset": dataset,
                "video_id": row["video_id"],
                "clip_index": 0,
                "label": row["label"],
                "probability": row["probability"],
                **({"method": row["method"]} if row.get("method") else {}),
            }
            for row in video_rows
        ]
    clip = compute(
        [int(row["label"]) for row in clip_rows],
        [float(row["probability"]) for row in clip_rows],
    )
    video = compute(
        [int(row["label"]) for row in video_rows],
        [float(row["probability"]) for row in video_rows],
    )
    per_video = {
        str(row["video_id"]): {
            "label": int(row["label"]),
            "probability": float(row["probability"]),
            "predicted": int(row.get("predicted", float(row["probability"]) >= 0.5)),
            **({"method": row["method"]} if row.get("method") else {}),
            "clip_probabilities": row.get("clip_probabilities", [row["probability"]]),
        }
        for row in video_rows
    }
    return {
        "dataset": dataset,
        "clip": clip,
        "video": video,
        "per_class": _per_class_from_rows(clip_rows, video_rows),
        "clips": clip_rows,
        "per_video": per_video,
    }


def write_experiment_results(
    *,
    results_dir: str | Path,
    dataset_results: dict[str, dict[str, Any]],
) -> Path:
    """Write per-dataset predictions/metrics and a micro-averaged overall folder.

    Overall metrics are recomputed from concatenated clip/video rows.
    Per-method tables stay in each dataset folder.
    """
    root = Path(results_dir)
    root.mkdir(parents=True, exist_ok=True)

    all_clip_labels: list[int] = []
    all_clip_scores: list[float] = []
    all_video_labels: list[int] = []
    all_video_scores: list[float] = []
    summary_rows: list[dict[str, Any]] = []
    per_dataset_short: dict[str, Any] = {}

    for dataset, result in sorted(dataset_results.items()):
        dataset_dir = root / dataset
        clip_rows = result.get("clips") or []
        video_rows = [
            {
                "dataset": dataset,
                "video_id": video_id,
                "label": row["label"],
                "probability": row["probability"],
                "predicted": row["predicted"],
                "method": row.get("method", ""),
                "clip_probabilities": row.get("clip_probabilities", []),
            }
            for video_id, row in sorted(result.get("per_video", {}).items())
        ]
        _write_jsonl(dataset_dir / "predictions.jsonl", video_rows)
        _write_jsonl(dataset_dir / "clip_predictions.jsonl", clip_rows)
        _write_json(dataset_dir / "metrics.json", _dataset_metrics(result))

        for row in clip_rows:
            all_clip_labels.append(int(row["label"]))
            all_clip_scores.append(float(row["probability"]))
        for row in video_rows:
            all_video_labels.append(int(row["label"]))
            all_video_scores.append(float(row["probability"]))

        video_m: Metrics = result["video"]
        clip_m: Metrics = result["clip"]
        per_dataset_short[dataset] = {
            "clip": _metrics_payload(clip_m),
            "video": _metrics_payload(video_m),
        }
        for split, metrics in (("clip", clip_m), ("video", video_m)):
            summary_rows.append(
                {
                    "scope": dataset,
                    "split": split,
                    **_metrics_payload(metrics),
                }
            )

    overall_clip = compute(all_clip_labels, all_clip_scores)
    overall_video = compute(all_video_labels, all_video_scores)
    overall_payload = {
        "n_datasets": len(dataset_results),
        "datasets": sorted(dataset_results),
        "n_videos": overall_video.n,
        "n_clips": overall_clip.n,
        "clip": _metrics_payload(overall_clip),
        "video": _metrics_payload(overall_video),
        "per_dataset": per_dataset_short,
    }
    _write_json(root / "overall" / "metrics.json", overall_payload)
    for split, metrics in (("clip", overall_clip), ("video", overall_video)):
        summary_rows.append({"scope": "overall", "split": split, **_metrics_payload(metrics)})
    _write_json(root / "overall" / "summary.json", {"rows": summary_rows})
    return root


def recompute_experiment_results(results_dir: str | Path) -> Path:
    """Rewrite metrics from saved prediction files under results_dir."""
    root = Path(results_dir)
    dataset_results: dict[str, dict[str, Any]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name == "overall":
            continue
        video_rows = _read_jsonl(child / "predictions.jsonl")
        if not video_rows:
            continue
        dataset_results[child.name] = result_from_prediction_rows(
            child.name,
            _read_jsonl(child / "clip_predictions.jsonl"),
            video_rows,
        )
    if not dataset_results:
        raise FileNotFoundError(f"no dataset predictions under {root}")
    return write_experiment_results(results_dir=root, dataset_results=dataset_results)
