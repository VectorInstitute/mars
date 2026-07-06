"""CLI entry point for Phase 1 eval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deepfake_moe.eval.dataset import EvalConfig, load_manifest_entries
from deepfake_moe.eval.metrics.report import generate_report
from deepfake_moe.eval.runner import run_eval
from deepfake_moe.models.base import Predictor
from deepfake_moe.models.config import Config
from deepfake_moe.models.registry import load_model
from deepfake_moe.models.vllm_config import VllmConfig
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Phase 1 deepfake eval (Qwen3-Omni)")
    p.add_argument("--eval-config", default="configs/eval.yaml")
    p.add_argument("--models-config", default="configs/models.yaml", help="Model registry YAML")
    p.add_argument("--model", default=None, help="Model name in the registry (default: eval config `model`)")
    p.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment label; results go to <output-dir>/<experiment>/<model> (default: eval config)",
    )
    p.add_argument("--output-dir", default=None, help="Override eval config output_dir")
    p.add_argument("--dry-run", action="store_true", help="5 clips per dataset")
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run only the clips that failed in a previous run of this experiment/model",
    )
    p.add_argument(
        "--metrics-only",
        action="store_true",
        help="Skip inference: aggregate existing shard predictions and (re)compute metrics",
    )
    p.add_argument("--num-shards", type=int, default=1, help="Total number of deterministic eval shards")
    p.add_argument("--shard-id", type=int, default=0, help="Current shard id in [0, num_shards)")
    return p.parse_args()


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _record_key(record: dict) -> tuple[str, str]:
    return str(record.get("dataset", "")), str(record.get("clip_id", ""))


def _dedup(rows: list[dict]) -> list[dict]:
    """Keep the first record per (dataset, clip_id), sorted for stable output."""
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows:
        by_key.setdefault(_record_key(row), row)
    return [by_key[key] for key in sorted(by_key)]


def _metrics_only(eval_cfg: EvalConfig, base_dir: Path) -> None:
    """Aggregate shard predictions (or a single run) and write metrics + report.

    Each shard writes its own ``shard_*/predictions.jsonl`` (safe under parallel
    Slurm tasks); this collects them into one deduplicated ``predictions.jsonl``
    at ``base_dir`` and regenerates the report — no model is loaded.
    """
    shard_files = sorted(base_dir.glob("shard_*/predictions.jsonl"))
    sources = shard_files or [base_dir / "predictions.jsonl"]
    sources = [p for p in sources if p.exists()]
    if not sources:
        raise FileNotFoundError(f"No predictions found under {base_dir} (run eval first)")

    rows: list[dict] = []
    for path in sources:
        rows.extend(_read_jsonl(path))
    records = _dedup(rows)
    logger.info("Aggregated %d rows -> %d unique from %d file(s)", len(rows), len(records), len(sources))

    _write_jsonl(base_dir / "predictions.jsonl", records)
    generate_report(records, eval_cfg.temporal_datasets, base_dir)
    logger.info("Metrics written to %s", base_dir)


def _retry_failed(
    model: Predictor,
    eval_cfg: EvalConfig,
    model_cfg: Config | VllmConfig,
    base_dir: Path,
) -> None:
    preds_path = base_dir / "predictions.jsonl"
    if not preds_path.exists():
        raise FileNotFoundError(f"No {preds_path} (run eval, or --metrics-only for a sharded run, before retrying)")
    existing = _read_jsonl(preds_path)
    failed_keys = {_record_key(r) for r in existing if not r.get("parse_ok", False)}
    if not failed_keys:
        logger.info("No failed clips to retry in %s", preds_path)
        return

    logger.info("Retrying %d failed clip(s) from %s", len(failed_keys), preds_path)
    entries = load_manifest_entries(eval_cfg, num_shards=1, shard_id=0)
    retry_entries = [e for e in entries if (str(e.get("dataset", "")), str(e.get("clip_id", ""))) in failed_keys]
    if not retry_entries:
        logger.warning("Failed clips are no longer in the sampled manifest; nothing to retry")
        return

    new_records = run_eval(model, retry_entries, eval_cfg, model_cfg, base_dir / "retry_predictions.jsonl")

    by_key = {_record_key(r): r for r in existing}
    for record in new_records:
        by_key[_record_key(record)] = record
    merged = [by_key[k] for k in sorted(by_key)]

    _write_jsonl(preds_path, merged)
    (base_dir / "retry_predictions.jsonl").unlink(missing_ok=True)
    generate_report(merged, eval_cfg.temporal_datasets, base_dir)
    logger.info("Retry complete. Updated %s", preds_path)


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    eval_cfg = EvalConfig.from_yaml(args.eval_config)

    if args.output_dir:
        eval_cfg.output_dir = args.output_dir

    model_name = args.model or eval_cfg.model
    experiment_name = args.experiment_name or eval_cfg.experiment_name
    base_dir = Path(eval_cfg.output_dir) / experiment_name / model_name
    logger.info("Experiment : %s   Model: %s", experiment_name, model_name)

    # Metrics-only skips inference entirely (no model load).
    if args.metrics_only:
        _metrics_only(eval_cfg, base_dir)
        return

    if args.dry_run:
        eval_cfg.num_clips_per_dataset = 5
        logger.info("Dry run — 5 clips per dataset")

    model, model_cfg = load_model(model_name, args.models_config)
    logger.info("Backend    : %s", getattr(model_cfg, "backend", "hf"))

    if args.retry_failed:
        _retry_failed(model, eval_cfg, model_cfg, base_dir)
        return

    output_dir = base_dir / f"shard_{args.shard_id:03d}" if args.num_shards > 1 else base_dir
    preds_path = output_dir / "predictions.jsonl"

    logger.info("Output dir : %s", output_dir)
    logger.info("Shard      : %d/%d", args.shard_id, args.num_shards)

    entries = load_manifest_entries(eval_cfg, num_shards=args.num_shards, shard_id=args.shard_id)
    records = run_eval(model, entries, eval_cfg, model_cfg, preds_path)
    generate_report(records, eval_cfg.temporal_datasets, output_dir)

    logger.info("Done. Results in %s", output_dir)


if __name__ == "__main__":
    main()
