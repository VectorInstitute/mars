"""GenVideo test-only manifest.

Real mp4s under real_dir (label 0, method real). Fake mp4s under
fake_dir/<generator> (label 1, method is the generator folder). Non-mp4 files
are ignored. When max_videos_per_generator is set, each generator is capped
and reals are kept to match the number of fakes actually taken.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.config import FAKE, REAL
from mars.data.genvideo.config import GenvideoConfig
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
        raise FileNotFoundError(f"GenVideo folder missing: {directory}")
    found: list[tuple[str, Path, str, int]] = []
    for path in sorted(directory.glob("*.mp4")):
        video_id = f"{method}/{path.name}"
        found.append((video_id, path, method, label))
    return found


def scan(cfg: GenvideoConfig) -> list[tuple[str, Path, str, int]]:
    """Discover (video_id, path, method, label) for reals and listed generators."""
    root = Path(cfg.root)
    found = _scan_folder(root / cfg.real_dir, "real", REAL)
    fake_root = root / cfg.fake_dir
    if not fake_root.is_dir():
        raise FileNotFoundError(f"GenVideo folder missing: {fake_root}")
    for generator in cfg.generators:
        found.extend(_scan_folder(fake_root / generator, generator, FAKE))
    return found


def limit_entries(
    cfg: GenvideoConfig,
    entries: list[tuple[str, Path, str, int]],
) -> list[tuple[str, Path, str, int]]:
    """Cap each fake generator, then keep as many reals as fakes kept.

    When max_videos_per_generator is unset, return entries unchanged.
    """
    if not cfg.max_videos_per_generator:
        return entries
    reals = [entry for entry in entries if entry[3] == REAL]
    fakes: list[tuple[str, Path, str, int]] = []
    per_generator: dict[str, int] = {}
    for entry in entries:
        if entry[3] != FAKE:
            continue
        method = entry[2]
        if per_generator.get(method, 0) < cfg.max_videos_per_generator:
            per_generator[method] = per_generator.get(method, 0) + 1
            fakes.append(entry)
    return reals[: min(len(reals), len(fakes))] + fakes


def build(cfg: GenvideoConfig, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. probe_frames(path) counts decodable frames."""
    entries = limit_entries(cfg, scan(cfg))
    if not entries:
        raise FileNotFoundError(f"no mp4 files found under {cfg.root}")

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
    lines = ["", f"{'split':<8}{'method':<16}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 51)
    seen: dict[tuple[str, str, int], list[int]] = {}
    for record in records:
        seen.setdefault((record.split, str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (split, method, label), frames in sorted(seen.items()):
        lines.append(f"{split:<8}{method:<16}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 51)
    n_real = sum(1 for record in records if record.label == REAL)
    lines.append(f"{'test':<8}{len(records):>5} videos  ({n_real} real / {len(records) - n_real} fake)")
    return "\n".join(lines)


def ensure_manifest(
    cfg: GenvideoConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the GenVideo manifest CSV if it does not already exist.

    Rank 0 writes; other ranks wait on barrier. Frame probing runs after the
    per-generator cap so unused videos are not decoded.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no GenVideo manifest at %s - scanning %s", path, cfg.root)
        entries = limit_entries(cfg, scan(cfg))
        with ThreadPoolExecutor(max_workers=16) as pool:
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
            f"GenVideo manifest missing at {path}. Build it with: "
            "python -m mars.data.genvideo --config configs/datasets/genvideo.yaml"
        )
    return path


def load_records(
    cfg: GenvideoConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
