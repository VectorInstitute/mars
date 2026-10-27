"""GTA-V (SAIL-VOS-3D) manifest.

Each listed clip is {root}/{seq}/images (numbered .bmp frames).
split_train.txt is train; split_val.txt is test. All clips are fake.
Clips whose images folder is not on disk yet are skipped.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.config import FAKE, REAL
from mars.data.gtav.config import GtavConfig
from mars.data.records import VideoRecord


MANIFEST_FIELDS = [
    "video_id",
    "path",
    "method",
    "label",
    "n_frames",
    "split",
]


def read_split(path: str | Path) -> list[str]:
    """Clip names from a SAIL-VOS split file (one per line)."""
    names: list[str] = []
    for line in Path(path).read_text().splitlines():
        name = line.strip()
        if name:
            names.append(name)
    return names


def scan(cfg: GtavConfig) -> list[tuple[str, Path, str, int, str]]:
    """Discover (video_id, images_dir, method, label, split) from split files."""
    root = Path(cfg.root)
    split_dir = root / cfg.split_dir
    found: list[tuple[str, Path, str, int, str]] = []
    for split, filename in (("train", cfg.train_file), ("test", cfg.test_file)):
        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"GTA-V split file missing: {path}")
        for name in read_split(path):
            images = root / name / cfg.images_dir
            if not images.is_dir():
                continue
            found.append((f"gtav/{name}", images, "gtav", FAKE, split))
    return found


def build(cfg: GtavConfig, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. probe_frames(path) counts frames in images/."""
    entries = scan(cfg)
    if not entries:
        raise FileNotFoundError(f"no clips found under {cfg.root}")

    if cfg.max_videos_per_class:
        limited: list[tuple[str, Path, str, int, str]] = []
        per_split: dict[str, int] = {}
        for entry in entries:
            split = entry[4]
            if per_split.get(split, 0) < cfg.max_videos_per_class:
                per_split[split] = per_split.get(split, 0) + 1
                limited.append(entry)
        entries = limited

    records: list[VideoRecord] = []
    for video_id, path, method, label, split in entries:
        n_frames = probe_frames(path)
        if n_frames <= 0:
            continue
        records.append(
            VideoRecord(
                video_id=video_id,
                path=str(path),
                label=label,
                n_frames=n_frames,
                split=split,
                extra={"method": method},
            )
        )
    if not records:
        raise FileNotFoundError(f"no decodable clips under {cfg.root}")
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
    """Human-readable split counts."""
    lines = ["", f"{'split':<8}{'method':<12}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 47)
    seen: dict[tuple[str, str, int], list[int]] = {}
    for record in records:
        seen.setdefault((record.split, str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (split, method, label), frames in sorted(seen.items()):
        lines.append(f"{split:<8}{method:<12}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 47)
    n_real = sum(1 for record in records if record.label == REAL)
    lines.append(f"{'total':<8}{len(records):>5} videos  ({n_real} real / {len(records) - n_real} fake)")
    return "\n".join(lines)


def ensure_manifest(
    cfg: GtavConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the GTA-V manifest CSV if it does not already exist.

    Rank 0 writes; other ranks wait on barrier.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no GTA-V manifest at %s - scanning %s", path, cfg.root)
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
            f"GTA-V manifest missing at {path}. Build it with: "
            "python -m mars.data.gtav --config configs/datasets/gtav.yaml"
        )
    return path


def load_records(
    cfg: GtavConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
