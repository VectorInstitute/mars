"""Celeb-DF v2 official test-list manifest.

Uses ``List_of_testing_videos.txt`` only (518 videos). The full tree is not
ingested. Official list column is 0=fake / 1=real; this repo uses REAL=0 /
FAKE=1. Folder names are the source of truth; the inverted official column
must agree.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.celebdf.config import CelebdfConfig
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

FOLDER_TO_LABEL = {
    "Celeb-synthesis": FAKE,
    "Celeb-real": REAL,
    "YouTube-real": REAL,
}


def official_to_repo_label(official: int) -> int:
    """Map the official list column (0=fake, 1=real) to ``REAL`` / ``FAKE``."""
    if official == 0:
        return FAKE
    if official == 1:
        return REAL
    raise ValueError(f"official Celeb-DF label must be 0 or 1, got {official}")


def label_from_folder(method: str) -> int:
    """Map a Celeb-DF source folder to ``REAL`` / ``FAKE``."""
    if method not in FOLDER_TO_LABEL:
        raise ValueError(f"unknown Celeb-DF folder {method!r}; expected one of {sorted(FOLDER_TO_LABEL)}")
    return FOLDER_TO_LABEL[method]


def repo_label(method: str, official: int, rel: str = "") -> int:
    """Folder label, asserted against the inverted official column."""
    folder = label_from_folder(method)
    mapped = official_to_repo_label(official)
    if folder != mapped:
        where = rel or method
        raise ValueError(f"{where}: official list label {official} maps to {mapped} but folder {method!r} is {folder}")
    return folder


def parse_list_line(line: str) -> tuple[int, str] | None:
    """Parse one official list row into ``(official_label, relative_path)``.

    Blank lines are skipped (``None``).
    """
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"malformed Celeb-DF list line: {line!r}")
    return int(parts[0]), parts[1]


def parse_list(path: str | Path) -> list[tuple[int, str]]:
    """Read official ``(label, relative_path)`` rows from ``path``."""
    rows: list[tuple[int, str]] = []
    for line in Path(path).read_text().splitlines():
        parsed = parse_list_line(line)
        if parsed is not None:
            rows.append(parsed)
    return rows


def list_path(cfg: CelebdfConfig) -> Path:
    """Resolve ``cfg.list_file`` against ``cfg.root`` when it is relative."""
    path = Path(cfg.list_file)
    if path.is_absolute():
        return path
    return Path(cfg.root) / path


def scan(cfg: CelebdfConfig) -> list[tuple[str, Path, str, int]]:
    """Discover ``(video_id, path, method, label)`` for the official test list.

    ``video_id`` is the relative path (e.g. ``Celeb-synthesis/id1_id0_0007.mp4``).
    Missing listed files raise.
    """
    path = list_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Celeb-DF test list missing: {path}")
    found: list[tuple[str, Path, str, int]] = []
    for official, rel in parse_list(path):
        video_path = Path(cfg.root) / rel
        if not video_path.is_file():
            raise FileNotFoundError(f"Celeb-DF listed video missing: {video_path}")
        method = rel.split("/", 1)[0]
        label = repo_label(method, official, rel)
        found.append((rel, video_path, method, label))
    return found


def build(cfg: CelebdfConfig, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. ``probe_frames(path) -> int`` counts decodable frames."""
    entries = scan(cfg)
    if not entries:
        raise FileNotFoundError(f"no videos found in Celeb-DF list {list_path(cfg)}")

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
        raise FileNotFoundError(f"no decodable videos under {cfg.root} from {list_path(cfg)}")
    return records


def write(records: list[VideoRecord], path: str | Path) -> None:
    """Write ``records`` as CSV."""
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
    lines = ["", f"{'split':<8}{'method':<20}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 55)
    seen: dict[tuple[str, str, int], list[int]] = {}
    for record in records:
        seen.setdefault((record.split, str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (split, method, label), frames in sorted(seen.items()):
        lines.append(f"{split:<8}{method:<20}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 55)
    n_real = sum(1 for r in records if r.label == REAL)
    lines.append(f"{'test':<8}{len(records):>5} videos  ({n_real} real / {len(records) - n_real} fake)")
    return "\n".join(lines)


def ensure_manifest(
    cfg: CelebdfConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the Celeb-DF manifest CSV if it does not already exist.

    Rank 0 writes; other ranks wait on ``barrier``.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no Celeb-DF manifest at %s - reading list %s", path, list_path(cfg))
        with ThreadPoolExecutor(max_workers=16) as pool:
            entries = scan(cfg)
            counts = dict(
                zip(
                    (e[1] for e in entries),
                    pool.map(count_frames, (e[1] for e in entries)),
                )
            )
        records = build(cfg, lambda p: counts.get(Path(p), 0))
        write(records, path)
        logger.info("%s", summarise(records))
    if barrier is not None:
        barrier()
    if not path.exists():
        raise FileNotFoundError(
            f"Celeb-DF manifest missing at {path}. Build it with: "
            "python -m mars.data.celebdf --config configs/datasets/celebdf.yaml"
        )
    return path


def load_records(
    cfg: CelebdfConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
