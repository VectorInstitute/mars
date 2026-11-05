"""Faithfulness runs for MARS.

Test A (splice): spliced FF++ clips are scored once; the fused margin is split exactly
over frames and patches (models/mars/attribution.py), and the frame scores are compared
with the known manipulated frames.

Test B (deletion): on sampled test clips, the frames or patches ranked highest by the
same split are removed, and the change in P(fake) and in AUROC is compared with removing
as many random frames or patches.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from mars.config import load_yaml
from mars.data.factory import build_split_datasets
from mars.eval.faithfulness.deletion import blank_patches, choose_frames, remove_frames
from mars.eval.faithfulness.splice import (
    SplicedClips,
    SpliceSpec,
    aligned_pairs,
    ffpp_test_pairs,
    splice_specs,
)
from mars.eval.summary import load_trainable_weights
from mars.models.mars.attribution import attribute
from mars.models.mars.model import Mars
from mars.utils.logging import get_logger


# The run fails if the parts and the model's margin differ by more than this.
SUM_TOLERANCE = 1e-3
# Source frames used for the alignment check of each pair.
ALIGN_FRAMES = [0, 40, 80, 126]
# Clips whose full frame maps are stored, for figures.
N_MAPS = 40
# Test B: corpora, clips per class and corpus, and removal sizes.
DELETION_CORPORA = ("ffpp", "celebdf", "dftimit", "uadfv", "hififace_paired", "genvideo", "df26")
DELETION_CLIPS = 100
FRAME_KS = (1, 2, 4, 8, 16)
PATCH_KS = (8, 16, 32, 64)


def build_mars(detector: Any, checkpoint: str, device: torch.device) -> Mars:
    """Build the trained MARS network in eval mode, in float32."""
    network = detector.build_network().to(device)
    load_trainable_weights(network, checkpoint, device)
    if not isinstance(network, Mars):
        raise ValueError("faithfulness needs two-path MARS")
    return network.float().eval()


def frame_localization(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Frame AUROC and top-1 hit of one clip's frame scores against its frame labels."""
    out = {"top1_hit": float(labels[int(np.argmax(scores))] == 1)}
    if 0 < labels.sum() < len(labels):
        out["frame_auroc"] = float(roc_auc_score(labels, scores))
    return out


def run_splice(
    detector: Any, checkpoint: str, out_dir: Path, batch_size: int = 4, workers: int = 8, limit: int | None = None
) -> dict[str, Any]:
    """Test A: score every spliced FF++ test clip and summarise the localization.

    Parameters
    ----------
    detector : BaseDetector
        MARS detector built from its config.
    checkpoint : str
        Trained checkpoint.
    out_dir : Path
        Where clips.jsonl, maps.npz, rejected.json and summary.json go.
    limit : int, optional
        Score only the first limit pairs (a smoke run).

    Returns
    -------
    dict
        The summary also written to summary.json.
    """
    log = get_logger(rank=0)
    cfg = detector.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mars(detector, checkpoint, device)

    manifest = load_yaml("configs/datasets/ffpp.yaml")["manifest"]
    n_frames, stride = cfg.data.n_frames, cfg.data.frame_stride
    candidates = ffpp_test_pairs(manifest, n_frames * stride)
    pairs, rejected = aligned_pairs(candidates[:limit] if limit else candidates, ALIGN_FRAMES)
    log.info("splice: %d aligned pairs, %d rejected", len(pairs), len(rejected))
    specs = splice_specs(pairs, (16, 32), n_frames)
    loader = DataLoader(
        SplicedClips(specs, n_frames, stride, cfg.data.image_size), batch_size=batch_size, num_workers=workers
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    maps: dict[str, np.ndarray] = {}
    worst = 0.0
    for done, (frames, valid, labels, index) in enumerate(loader, start=1):
        parts = attribute(model, frames.to(device), valid.to(device))
        gap = float((parts.margin_from_parts - parts.margin).abs().max())
        worst = max(worst, gap)
        if gap > SUM_TOLERANCE:
            raise RuntimeError(f"frame contributions miss the margin by {gap:.2e}")
        for row in range(len(index)):
            spec = specs[int(index[row])]
            scores = parts.frame_scores[row].cpu().numpy()
            truth = labels[row].numpy()
            record = {
                "clip": int(index[row]),
                "original": spec.pair.original,
                "manipulated": spec.pair.manipulated,
                "method": spec.pair.method,
                "length": spec.length,
                "position": spec.position,
                "p_fake": float(torch.sigmoid(parts.margin[row])),
                "gate": float(parts.gate[row]),
                "frame_scores": scores.round(6).tolist(),
                "labels": truth.tolist(),
                **frame_localization(scores, truth),
            }
            records.append(record)
            if len(maps) < N_MAPS and spec.length in (16, 32):
                maps[f"clip{record['clip']}"] = parts.frame_maps[row].cpu().numpy().astype(np.float16)
        if done % 50 == 0:
            log.info("splice: %d / %d batches", done, len(loader))

    with open(out_dir / "clips.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    np.savez_compressed(out_dir / "maps.npz", **maps)  # type: ignore[arg-type]
    (out_dir / "rejected.json").write_text(json.dumps([[p.manipulated, d] for p, d in rejected], indent=1))
    summary = summarise_splice(records, len(pairs), len(rejected), worst)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("splice: %s", json.dumps(summary["overall"]))
    return summary


def summarise_splice(records: list[dict[str, Any]], n_pairs: int, n_rejected: int, worst: float) -> dict[str, Any]:
    """Mean frame AUROC, top-1 hit rate and its chance level, overall and per group."""

    def stats(rows: list[dict[str, Any]]) -> dict[str, float]:
        spliced = [r for r in rows if "frame_auroc" in r]
        chance = [r["length"] / len(r["labels"]) for r in spliced]
        return {
            "clips": len(spliced),
            "frame_auroc": float(np.mean([r["frame_auroc"] for r in spliced])),
            "top1_hit": float(np.mean([r["top1_hit"] for r in spliced])),
            "top1_chance": float(np.mean(chance)),
            "p_fake": float(np.mean([r["p_fake"] for r in spliced])),
        }

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        if "frame_auroc" in r:
            for key in (f"method={r['method']}", f"length={r['length']}", f"position={r['position']}"):
                groups[key].append(r)
    reference = {
        name: float(np.mean([r["p_fake"] for r in records if r["position"] == name])) for name in ("none", "all")
    }
    return {
        "pairs": n_pairs,
        "rejected_pairs": n_rejected,
        "max_sum_gap": worst,
        "overall": stats(records),
        "groups": {k: stats(v) for k, v in sorted(groups.items())},
        "p_fake_unspliced": reference,
    }


def deletion_clips(dataset: Any, predictions: Path, n_per_class: int, seed: int) -> list[int]:
    """Choose the clips used for deletion in one corpus.

    The first clip of each video, kept only if it has more genuine frames than the
    largest frame removal: n_per_class manipulated clips that the main run scored as
    fake (probability above 0.5) and n_per_class authentic clips, drawn with a fixed
    seed.
    """
    index = {(c.video_id, c.clip_index): i for i, c in enumerate(dataset.clips)}
    fake, real = [], []
    with open(predictions) as f:
        for line in f:
            row = json.loads(line)
            i = index.get((row["video_id"], row["clip_index"]))
            if row["clip_index"] != 0 or i is None or dataset.clips[i].n_valid <= max(FRAME_KS):
                continue
            if row["label"] == 1 and row["probability"] > 0.5:
                fake.append(i)
            elif row["label"] == 0:
                real.append(i)
    rng = np.random.default_rng(seed)

    def pick(pool: list[int]) -> list[int]:
        return sorted(int(i) for i in rng.choice(pool, min(n_per_class, len(pool)), replace=False))

    return pick(fake) + pick(real)


@torch.no_grad()
def _p_fake(model: Mars, frames: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """P(fake) of a batch of clips, in float32."""
    logits = model(frames, valid=valid).logits.float()
    return torch.sigmoid(logits[:, 1] - logits[:, 0])


def run_deletion(
    detector: Any,
    checkpoint: str,
    out_dir: Path,
    main_run: Path,
    corpora: tuple[str, ...] = DELETION_CORPORA,
    n_per_class: int = DELETION_CLIPS,
    seed: int = 0,
) -> dict[str, Any]:
    """Test B: remove the top-ranked or random frames and patches and rescore.

    Parameters
    ----------
    detector : BaseDetector
        MARS detector built from its config.
    checkpoint : str
        Trained checkpoint.
    out_dir : Path
        Where clips.jsonl and summary.json go.
    main_run : Path
        Results folder of the main run; its clip predictions choose the clips.
    corpora : tuple of str
        Test sets to use.
    n_per_class : int
        Manipulated and authentic clips per corpus.
    seed : int
        Seed of the clip choice and of the random removals.

    Returns
    -------
    dict
        The summary also written to summary.json.
    """
    log = get_logger(rank=0)
    cfg = detector.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mars(detector, checkpoint, device)
    datasets = build_split_datasets(cfg.data, need=("test",), test_names=list(corpora))["test"]
    grid = cfg.model.grid_side
    rng = torch.Generator().manual_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for corpus in corpora:
        dataset = datasets[corpus]
        chosen = deletion_clips(dataset, main_run / corpus / "clip_predictions.jsonl", n_per_class, seed)
        log.info("deletion: %s, %d clips", corpus, len(chosen))
        for done, i in enumerate(chosen, start=1):
            sample = dataset[i]
            frames = sample.frames.to(device)
            valid = (torch.arange(frames.shape[0]) < sample.n_valid).to(device)
            parts = attribute(model, frames[None], valid[None])
            n_valid = int(valid.sum())
            frame_rank = parts.frame_scores[0, :n_valid].argsort(descending=True).cpu()
            patch_rank = parts.frame_maps[0].sum(dim=0).argsort(descending=True).cpu()

            variants, keys = [], []
            for k in FRAME_KS:
                for kind, drop in (("ranked", frame_rank[:k]), ("random", torch.randperm(n_valid, generator=rng)[:k])):
                    variants.append(remove_frames(frames, valid, drop.to(device)))
                    keys.append(("frames", kind, k))
            for k in PATCH_KS:
                for kind, patches in (
                    ("ranked", patch_rank[:k]),
                    ("random", torch.randperm(grid * grid, generator=rng)[:k]),
                ):
                    variants.append((blank_patches(frames, patches, grid), valid))
                    keys.append(("patches", kind, k))
            scores = []
            for start in range(0, len(variants), 6):
                chunk = variants[start : start + 6]
                scores.append(
                    _p_fake(model, torch.stack([f for f, _ in chunk]), torch.stack([v for _, v in chunk])).cpu()
                )
            p = torch.cat(scores)
            records.append(
                {
                    "corpus": corpus,
                    "video_id": dataset.clips[i].video_id,
                    "label": int(sample.label),
                    "p_fake": float(torch.sigmoid(parts.margin[0])),
                    "edits": [
                        {"target": t, "kind": kd, "k": k, "p_fake": float(v)}
                        for (t, kd, k), v in zip(keys, p, strict=True)
                    ],
                }
            )
            if done % 25 == 0:
                log.info("deletion: %s %d / %d", corpus, done, len(chosen))

    with open(out_dir / "clips.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    summary = summarise_deletion(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("deletion: done, %d clips", len(records))
    return summary


def summarise_deletion(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean P(fake) of manipulated clips and AUROC, per corpus, target, kind and k."""
    out: dict[str, Any] = {}
    for corpus in sorted({r["corpus"] for r in records}):
        rows = [r for r in records if r["corpus"] == corpus]
        labels = np.array([r["label"] for r in rows])
        clean = np.array([r["p_fake"] for r in rows])
        table: dict[str, Any] = {"clean": {"p_fake_manipulated": float(clean[labels == 1].mean())}}
        if 0 < labels.sum() < len(labels):
            table["clean"]["auroc"] = float(roc_auc_score(labels, clean))
        for edit in rows[0]["edits"]:
            key = f"{edit['target']}/{edit['kind']}/k={edit['k']}"
            p = np.array(
                [
                    next(
                        e["p_fake"]
                        for e in r["edits"]
                        if (e["target"], e["kind"], e["k"]) == (edit["target"], edit["kind"], edit["k"])
                    )
                    for r in rows
                ]
            )
            entry = {"p_fake_manipulated": float(p[labels == 1].mean())}
            if 0 < labels.sum() < len(labels):
                entry["auroc"] = float(roc_auc_score(labels, p))
            table[key] = entry
        out[corpus] = table
    return out


def run_temporal(
    detector: Any, checkpoint: str, out_dir: Path, seed: int = 0, limit: int | None = None
) -> dict[str, Any]:
    """Run Test B2: remove ranked, random or manipulated frames from spliced clips.

    Parameters
    ----------
    detector : BaseDetector
        MARS detector built from its config.
    checkpoint : str
        Trained checkpoint.
    out_dir : Path
        Where clips.jsonl and summary.json go.
    seed : int
        Seed of the random and oracle choices.
    limit : int, optional
        Use only the first limit pairs (a smoke run).

    Returns
    -------
    dict
        The summary also written to summary.json.
    """
    log = get_logger(rank=0)
    cfg = detector.cfg
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_mars(detector, checkpoint, device)
    manifest = load_yaml("configs/datasets/ffpp.yaml")["manifest"]
    n_frames, stride = cfg.data.n_frames, cfg.data.frame_stride
    candidates = ffpp_test_pairs(manifest, n_frames * stride)
    pairs, rejected = aligned_pairs(candidates[:limit] if limit else candidates, ALIGN_FRAMES)
    log.info("temporal: %d aligned pairs, %d rejected", len(pairs), len(rejected))
    specs = [SpliceSpec(p, length, "middle") for p in pairs for length in (16, 32)]
    specs += [SpliceSpec(p, 0, "none") for p in pairs]
    clips = SplicedClips(specs, n_frames, stride, cfg.data.image_size)
    generator = torch.Generator().manual_seed(seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    authentic: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    for done, i in enumerate(range(len(specs)), start=1):
        spec = specs[i]
        frames, valid, labels, _ = clips[i]
        frames, valid = frames.to(device), valid.to(device)
        if spec.length == 0:
            authentic[spec.pair.manipulated] = float(_p_fake(model, frames[None], valid[None])[0])
            continue
        parts = attribute(model, frames[None], valid[None])
        scores = parts.frame_scores[0].cpu()
        edits = []
        for k in (spec.length // 4, spec.length // 2, spec.length):
            for kind in ("ranked", "random", "oracle"):
                drop = choose_frames(scores, labels, k, kind, generator)
                edited, edited_valid = remove_frames(frames, valid, drop.to(device))
                edits.append(
                    {
                        "kind": kind,
                        "k": k,
                        "p_fake": float(_p_fake(model, edited[None], edited_valid[None])[0]),
                        "share_manipulated": float(labels[drop].float().mean()),
                    }
                )
        records.append(
            {
                "manipulated": spec.pair.manipulated,
                "method": spec.pair.method,
                "length": spec.length,
                "p_fake": float(torch.sigmoid(parts.margin[0])),
                "edits": edits,
            }
        )
        if done % 200 == 0:
            log.info("temporal: %d / %d clips", done, len(specs))
    for r in records:
        r["p_fake_authentic"] = authentic[r["manipulated"]]

    with open(out_dir / "clips.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in records)
    summary = summarise_temporal(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    log.info("temporal: %s", json.dumps(summary))
    return summary


def summarise_temporal(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Average P(fake) and share of manipulated frames removed per length, k, choice."""
    out: dict[str, Any] = {}
    for length in sorted({r["length"] for r in records}):
        rows = [r for r in records if r["length"] == length]
        table: dict[str, Any] = {
            "clips": len(rows),
            "p_fake_spliced": float(np.mean([r["p_fake"] for r in rows])),
            "p_fake_authentic": float(np.mean([r["p_fake_authentic"] for r in rows])),
        }
        for edit in rows[0]["edits"]:
            key = f"{edit['kind']}/k={edit['k']}"
            matched = [e for r in rows for e in r["edits"] if (e["kind"], e["k"]) == (edit["kind"], edit["k"])]
            table[key] = {
                "p_fake": float(np.mean([e["p_fake"] for e in matched])),
                "share_manipulated": float(np.mean([e["share_manipulated"] for e in matched])),
            }
        out[f"length={length}"] = table
    return out
