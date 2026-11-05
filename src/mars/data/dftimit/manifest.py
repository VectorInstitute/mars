"""DeepfakeTIMIT test-only manifest.

Real: one sentence directory of numbered JPEGs is one video. Fake: one AVI
under higher_quality or lower_quality. Matching real to fake uses subject and
sentence; the identity after video- is the inserted face only. Ignore head,
head2, head3, wav, and dircksum files. VidTIMIT-Real-unused is not scanned.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.config import FAKE, REAL
from mars.data.dftimit.config import DftimitConfig
from mars.data.records import VideoRecord


MANIFEST_FIELDS = [
    "video_id",
    "path",
    "method",
    "label",
    "n_frames",
    "split",
    "subject",
    "sentence",
    "inserted",
]

SKIP_SENTENCES = {"head", "head2", "head3"}
FAKE_NAME = re.compile(r"^(.+)-video-(.+)\.avi$")


def parse_fake_name(name: str) -> tuple[str, str]:
    """Return (sentence, inserted identity) from an AVI filename."""
    match = FAKE_NAME.match(name)
    if match is None:
        raise ValueError(f"unrecognized DeepfakeTIMIT fake name: {name}")
    return match.group(1), match.group(2)


def extra_fields(
    *,
    method: str,
    subject: str,
    sentence: str,
    inserted: str = "",
) -> dict[str, str]:
    """Per-video extras used by eval breakdowns."""
    return {
        "method": method,
        "subject": subject,
        "sentence": sentence,
        "inserted": inserted,
    }


def scan_real(cfg: DftimitConfig) -> list[tuple[str, Path, dict[str, str], int]]:
    """Discover real sentence directories under real_dir."""
    root = Path(cfg.root) / cfg.real_dir
    if not root.is_dir():
        raise FileNotFoundError(f"DeepfakeTIMIT real root missing: {root}")
    found: list[tuple[str, Path, dict[str, str], int]] = []
    for subject_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        video_dir = subject_dir / "video"
        if not video_dir.is_dir():
            continue
        for sentence_dir in sorted(p for p in video_dir.iterdir() if p.is_dir()):
            if sentence_dir.name in SKIP_SENTENCES:
                continue
            extra = extra_fields(
                method="real",
                subject=subject_dir.name,
                sentence=sentence_dir.name,
            )
            video_id = f"real/{subject_dir.name}/{sentence_dir.name}"
            found.append((video_id, sentence_dir, extra, REAL))
    return found


def scan_fake(cfg: DftimitConfig) -> list[tuple[str, Path, dict[str, str], int]]:
    """Discover HQ and LQ AVI files under fake_dir."""
    root = Path(cfg.root) / cfg.fake_dir
    if not root.is_dir():
        raise FileNotFoundError(f"DeepfakeTIMIT fake root missing: {root}")
    found: list[tuple[str, Path, dict[str, str], int]] = []
    for quality in cfg.qualities:
        quality_dir = root / quality
        if not quality_dir.is_dir():
            raise FileNotFoundError(f"DeepfakeTIMIT quality folder missing: {quality_dir}")
        for subject_dir in sorted(p for p in quality_dir.iterdir() if p.is_dir()):
            for avi in sorted(subject_dir.glob("*.avi")):
                sentence, inserted = parse_fake_name(avi.name)
                extra = extra_fields(
                    method=quality,
                    subject=subject_dir.name,
                    sentence=sentence,
                    inserted=inserted,
                )
                video_id = f"{quality}/{subject_dir.name}/{avi.stem}"
                found.append((video_id, avi, extra, FAKE))
    return found


def scan(cfg: DftimitConfig) -> list[tuple[str, Path, dict[str, str], int]]:
    """Discover (video_id, path, extra, label) for the full test set."""
    return scan_real(cfg) + scan_fake(cfg)


def build(cfg: DftimitConfig, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. probe_frames(path) counts frames in a file or directory."""
    entries = scan(cfg)
    if not entries:
        raise FileNotFoundError(f"no videos found under {cfg.root}")

    if cfg.max_videos_per_class:
        limited: list[tuple[str, Path, dict[str, str], int]] = []
        per_method: dict[str, int] = {}
        for entry in entries:
            method = entry[2]["method"]
            if per_method.get(method, 0) < cfg.max_videos_per_class:
                per_method[method] = per_method.get(method, 0) + 1
                limited.append(entry)
        entries = limited

    records: list[VideoRecord] = []
    for video_id, path, extra, label in entries:
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
                extra=dict(extra),
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
                extra={
                    "method": row["method"],
                    "subject": row.get("subject", ""),
                    "sentence": row.get("sentence", ""),
                    "inserted": row.get("inserted", ""),
                },
            )
            for row in csv.DictReader(handle)
        ]


def summarise(records: list[VideoRecord]) -> str:
    """Human-readable method counts."""
    lines = ["", f"{'split':<8}{'method':<20}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 55)
    seen: dict[tuple[str, str, int], list[int]] = {}
    for record in records:
        seen.setdefault((record.split, str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (split, method, label), frames in sorted(seen.items()):
        lines.append(f"{split:<8}{method:<20}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 55)
    n_real = sum(1 for record in records if record.label == REAL)
    lines.append(f"{'test':<8}{len(records):>5} videos  ({n_real} real / {len(records) - n_real} fake)")
    return "\n".join(lines)


def ensure_manifest(
    cfg: DftimitConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the DeepfakeTIMIT manifest CSV if it does not already exist.

    Rank 0 writes; other ranks wait on barrier.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no DeepfakeTIMIT manifest at %s - scanning %s", path, cfg.root)
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
            f"DeepfakeTIMIT manifest missing at {path}. Build it with: "
            "python -m mars.data.dftimit --config configs/datasets/dftimit.yaml"
        )
    return path


def load_records(
    cfg: DftimitConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
