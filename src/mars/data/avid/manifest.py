"""AVID test-only manifest.

Two folders of mp4 files: real_dir is real (label 0), fake_dir is fake (label 1).
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.avid.config import AvidConfig
from mars.data.config import FAKE, REAL
from mars.data.records import VideoRecord


MANIFEST_FIELDS = [
    "video_id",
    "path",
    "method",
    "label",
    "n_frames",
    "split",
]


def _scan_folder(directory: Path, method: str, label: int) -> list[tuple[str, Path, str, int]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"AVID folder missing: {directory}")
    found: list[tuple[str, Path, str, int]] = []
    for path in sorted(directory.glob("*.mp4")):
        video_id = f"{method}/{path.name}"
        found.append((video_id, path, method, label))
    return found


def scan(cfg: AvidConfig) -> list[tuple[str, Path, str, int]]:
    """Discover (video_id, path, method, label) under real_dir and fake_dir."""
    root = Path(cfg.root)
    return _scan_folder(root / cfg.real_dir, "real", REAL) + _scan_folder(root / cfg.fake_dir, "fake", FAKE)


def build(cfg: AvidConfig, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. probe_frames(path) counts decodable frames."""
    entries = scan(cfg)
    if not entries:
        raise FileNotFoundError(f"no mp4 files found under {cfg.root}")

    if cfg.max_videos_per_class:
        limited: list[tuple[str, Path, str, int]] = []
        per_method: dict[str, int] = {}
        for entry in entries:
            method = entry[2]
            if per_method.get(method, 0) < cfg.max_videos_per_class:
                per_method[method] = per_method.get(method, 0) + 1
                limited.append(entry)
        entries = limited

    records: list[VideoRecord] = []
    for video_id, path, method, label in entries:
        n_frames = probe_frames(path)
        if n_frames <= 0:
            continue
        records.append(
            VideoRecord(
                video_id=video_id,
                path=str(path),
                label=label,
                n_frames=n_frames,
                split="test",
                extra={"method": method},
            )
        )
    if not records:
        raise FileNotFoundError(f"no decodable videos under {cfg.root}")
    return records


def write(records: list[VideoRecord], path: str | Path) -> None:
    """Write records as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            row = {
                "video_id": record.video_id,
                "path": record.path,
                "label": record.label,
                "n_frames": record.n_frames,
                "split": record.split,
                **record.extra,
            }
            writer.writerow(row)


def read(path: str | Path) -> list[VideoRecord]:
    """Read a previously written manifest CSV."""
    with Path(path).open() as handle:
        return [
            VideoRecord(
                video_id=row["video_id"],
                path=row["path"],
                label=int(row["label"]),
                n_frames=int(row["n_frames"]),
                split=row["split"],
                extra={"method": row["method"]},
            )
            for row in csv.DictReader(handle)
        ]


def summarise(records: list[VideoRecord]) -> str:
    """Human-readable method counts."""
    lines = ["", f"{'split':<8}{'method':<12}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 47)
    seen: dict[tuple[str, str, int], list[int]] = {}
    for record in records:
        seen.setdefault((record.split, str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (split, method, label), frames in sorted(seen.items()):
        lines.append(f"{split:<8}{method:<12}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 47)
    n_real = sum(1 for record in records if record.label == REAL)
    lines.append(f"{'test':<8}{len(records):>5} videos  ({n_real} real / {len(records) - n_real} fake)")
    return "\n".join(lines)


def ensure_manifest(
    cfg: AvidConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the AVID manifest CSV if it does not already exist.

    Rank 0 writes; other ranks wait on barrier.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no AVID manifest at %s - scanning %s", path, cfg.root)
        with ThreadPoolExecutor(max_workers=16) as pool:
            entries = scan(cfg)
            counts = dict(
                zip(
                    (entry[1] for entry in entries),
                    pool.map(count_frames, (entry[1] for entry in entries)),
                )
            )
        records = build(cfg, lambda p: counts.get(Path(p), 0))
        write(records, path)
        logger.info("%s", summarise(records))
    if barrier is not None:
        barrier()
    if not path.exists():
        raise FileNotFoundError(
            f"AVID manifest missing at {path}. Build it with: "
            "python -m mars.data.avid --config configs/datasets/avid.yaml"
        )
    return path


def load_records(
    cfg: AvidConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
