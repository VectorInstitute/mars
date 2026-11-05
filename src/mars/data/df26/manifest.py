"""DF26 test-only manifest.

Videos are listed from the release's metadata CSV rather than a folder walk, so the
scored set is exactly the official one. Reals get method real; fakes get their
generator name, and only the configured generators are kept.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.config import FAKE, REAL
from mars.data.df26.config import Df26Config
from mars.data.records import VideoRecord


MANIFEST_FIELDS = [
    "video_id",
    "path",
    "method",
    "label",
    "n_frames",
    "split",
]


def scan(cfg: Df26Config) -> list[tuple[str, Path, str, int]]:
    """List (video_id, path, method, label) for every real and each listed generator."""
    root = Path(cfg.root)
    metadata = root / cfg.metadata
    if not metadata.is_file():
        raise FileNotFoundError(f"DF26 metadata missing: {metadata}")
    wanted = set(cfg.generators)
    found: list[tuple[str, Path, str, int]] = []
    with metadata.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["label"] == "real":
                method, label = "real", REAL
            elif row["label"] == "fake":
                method, label = row["generator"], FAKE
                if method not in wanted:
                    continue
            else:
                raise ValueError(f"unknown DF26 label {row['label']!r} for {row['video_path']}")
            found.append((row["video_path"], root / row["video_path"], method, label))
    return found


def build(cfg: Df26Config, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. probe_frames(path) counts decodable frames."""
    entries = scan(cfg)
    if not entries:
        raise FileNotFoundError(f"no videos listed in {Path(cfg.root) / cfg.metadata}")

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
            writer.writerow(
                {
                    "video_id": record.video_id,
                    "path": record.path,
                    "label": record.label,
                    "n_frames": record.n_frames,
                    "split": record.split,
                    **record.extra,
                }
            )


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
    lines = ["", f"{'method':<28}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 55)
    seen: dict[tuple[str, int], list[int]] = {}
    for record in records:
        seen.setdefault((str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (method, label), frames in sorted(seen.items()):
        lines.append(f"{method:<28}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 55)
    n_real = sum(1 for record in records if record.label == REAL)
    lines.append(f"{len(records)} videos  ({n_real} real / {len(records) - n_real} fake)")
    return "\n".join(lines)


def ensure_manifest(
    cfg: Df26Config,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the DF26 manifest CSV if it does not already exist.

    Rank 0 writes; other ranks wait on barrier.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no DF26 manifest at %s - scanning %s", path, cfg.root)
        paths = [entry[1] for entry in scan(cfg)]
        with ThreadPoolExecutor(max_workers=16) as pool:
            counts = dict(zip(paths, pool.map(count_frames, paths)))
        records = build(cfg, lambda p: counts.get(Path(p), 0))
        write(records, path)
        logger.info("%s", summarise(records))
    if barrier is not None:
        barrier()
    if not path.exists():
        raise FileNotFoundError(
            f"DF26 manifest missing at {path}. Build it with: "
            "python -m mars.data.df26 --config configs/datasets/df26.yaml"
        )
    return path


def load_records(
    cfg: Df26Config,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
