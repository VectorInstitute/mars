"""Per-dataset sampling for SFT training/validation entries.

Train sampling is proportionally stratified by joint AV label (real /
fake_video / fake_audio / fake_both): each stratum is sampled at
(approximately) its share of the pool, so a fixed ``train_samples_per_dataset``
draw can't accidentally miss an entire class due to random luck on a small
sample fraction. This preserves the pool's natural class balance rather than
rebalancing it — it guarantees coverage, not equal representation.

Validation sampling is plain uniform (matches ``eval.dataset``'s approach) —
stratification is for train sampling specifically.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

_JOINT_AV_CLASSES: dict[tuple[int, int], str] = {
    (0, 0): "real",
    (1, 0): "fake_video",
    (0, 1): "fake_audio",
    (1, 1): "fake_both",
}


def _joint_class(entry: dict) -> str:
    label = entry.get("label") or [0, 0]
    key = (int(label[0]), int(label[1]))
    return _JOINT_AV_CLASSES.get(key, "real")


def stratified_sample(pool: list[dict], n: int, *, seed: int) -> list[dict]:
    """Proportional stratified sample of ``n`` entries from ``pool`` by joint AV label.

    Each stratum's quota is ``round(n * stratum_size / pool_size)``, adjusted
    by largest-remainder so quotas sum exactly to ``n``. Entries within a
    stratum are drawn without replacement via a seeded RNG, so results are
    deterministic and reproducible.
    """
    if n >= len(pool):
        return list(pool)

    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = {}
    for idx, entry in enumerate(pool):
        by_class.setdefault(_joint_class(entry), []).append(idx)

    exact_quotas = {cls: n * len(idxs) / len(pool) for cls, idxs in by_class.items()}
    quotas = {cls: int(q) for cls, q in exact_quotas.items()}
    remainder = n - sum(quotas.values())
    by_fractional_part = sorted(exact_quotas, key=lambda cls: exact_quotas[cls] - quotas[cls], reverse=True)
    for cls in by_fractional_part[:remainder]:
        quotas[cls] += 1

    selected: list[int] = []
    for cls, idxs in by_class.items():
        k = min(quotas[cls], len(idxs))
        chosen = rng.choice(len(idxs), size=k, replace=False)
        selected.extend(idxs[i] for i in chosen)

    logger.info("stratified sample: pool=%d -> %d  quotas=%s", len(pool), len(selected), quotas)
    return [pool[i] for i in sorted(selected)]


def uniform_sample(pool: list[dict], n: int, *, seed: int) -> list[dict]:
    """Plain uniform sample of ``n`` entries from ``pool`` (no stratification)."""
    if n >= len(pool):
        return list(pool)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pool), size=n, replace=False)
    return [pool[i] for i in sorted(indices)]


def read_manifest_pool(manifest_dir: str, filenames: list[str]) -> list[dict]:
    """Read and concatenate one dataset's labeled manifest rows.

    Shared by :func:`load_split_entries` and ``train/avr_ssl/dataset.py``'s
    real-only loader, which filters this pool before sampling rather than
    after (see that module for why the order matters there).
    """
    manifest_root = Path(manifest_dir)
    pool: list[dict] = []
    for fname in filenames:
        fpath = manifest_root / fname
        if not fpath.exists():
            logger.warning("manifest not found, skipping: %s", fpath)
            continue
        with open(fpath) as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                entry = json.loads(text)
                if entry.get("label") is None:
                    continue
                pool.append(entry)
    return pool


def load_split_entries(
    manifest_dir: str,
    dataset_sources: dict[str, list[str]],
    samples_per_dataset: int,
    *,
    seed: int,
    stratify: bool,
) -> list[dict]:
    """Load and sample manifest entries per dataset for train/val (never blind).

    Mirrors ``eval.dataset.load_manifest_entries``'s file-reading logic, minus
    sharding/inference-only handling (not applicable to training), plus the
    stratification choice above.
    """
    all_entries: list[dict] = []

    for dataset, filenames in dataset_sources.items():
        pool = read_manifest_pool(manifest_dir, filenames)
        if not pool:
            logger.warning("No usable entries for dataset=%s", dataset)
            continue

        sample_fn = stratified_sample if stratify else uniform_sample
        sampled = sample_fn(pool, samples_per_dataset, seed=seed)
        logger.info("dataset=%-15s  pool=%6d  sampled=%d  stratified=%s", dataset, len(pool), len(sampled), stratify)
        all_entries.extend(sampled)

    logger.info("Total entries: %d", len(all_entries))
    return all_entries
