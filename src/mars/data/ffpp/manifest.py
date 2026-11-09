"""FaceForensics++ manifest with identity-safe splits.

Default: official FaceForensics++ ``train.json`` / ``val.json`` / ``test.json``
under ``src/mars/data/ffpp/splits`` (720 / 140 / 140 original IDs). Manipulated videos
are named ``<target>_<source>.mp4``; both IDs must land in the same split.

If those files are absent, connected components of shared source IDs are
assigned wholly to one split using ``split_fractions``.
"""

from __future__ import annotations

import csv
import json
import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mars.data.config import FAKE, REAL
from mars.data.ffpp.config import FfppConfig
from mars.data.records import VideoRecord


MANIFEST_FIELDS = [
    "video_id",
    "path",
    "method",
    "label",
    "compression",
    "n_frames",
    "split",
    "group",
]

PACKAGE_SPLITS = Path(__file__).resolve().parent / "splits"


def parse_ids(stem: str) -> tuple[str, ...]:
    """``'035_036'`` -> ``('035','036')``; ``'035'`` -> ``('035',)``."""
    return tuple(part for part in stem.split("_") if part)


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def scan(cfg: FfppConfig) -> list[tuple[str, Path, str, int]]:
    """Discover ``(stem, path, method, label)`` for the configured subsets."""
    root = Path(cfg.root)
    found: list[tuple[str, Path, str, int]] = []
    for source in cfg.originals:
        directory = root / "original_sequences" / source / cfg.compression / "videos"
        for path in sorted(directory.glob("*.mp4")):
            found.append((path.stem, path, source, REAL))
    for method in cfg.manipulations:
        directory = root / "manipulated_sequences" / method / cfg.compression / "videos"
        for path in sorted(directory.glob("*.mp4")):
            found.append((path.stem, path, method, FAKE))
    return found


def assign_groups(entries: list[tuple[str, Path, str, int]]) -> dict[str, int]:
    """Map each video stem to a connected-component id over shared source IDs."""
    dsu = _DisjointSet()
    for stem, _, _, _ in entries:
        ids = parse_ids(stem)
        for other in ids[1:]:
            dsu.union(ids[0], other)
        dsu.find(ids[0])
    roots = sorted({dsu.find(i) for i in dsu.parent})
    root_to_group = {root: i for i, root in enumerate(roots)}
    return {stem: root_to_group[dsu.find(parse_ids(stem)[0])] for stem, *_ in entries}


def split_groups(groups: set[int], fractions: list[float], seed: int) -> dict[int, str]:
    """Deterministically assign whole groups to train/val/test."""
    ordered = sorted(groups)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    if n >= 3:
        n_train = max(1, min(n_train, n - 2))
        n_val = max(1, min(n_val, n - n_train - 1))
    else:
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
    names = ["train"] * n_train + ["val"] * n_val
    names += ["test"] * (n - len(names))
    return dict(zip(ordered, names))


def resolve_splits_dir(cfg: FfppConfig) -> Path | None:
    """Directory of official ``train.json`` / ``val.json`` / ``test.json``, or None.

    Uses ``cfg.split_file`` when set. Otherwise uses the bundled package splits
    if ``train.json`` is present. Callers fall back to ``split_fractions``.
    """
    if cfg.split_file:
        return Path(cfg.split_file)
    if (PACKAGE_SPLITS / "train.json").is_file():
        return PACKAGE_SPLITS
    return None


def split_from_dir(directory: str | Path, entries: list[tuple[str, Path, str, int]]) -> dict[str, str]:
    """Read official FF++ ``train.json`` / ``val.json`` / ``test.json`` pair lists."""
    directory = Path(directory)
    id_to_split: dict[str, str] = {}
    for name in ("train", "val", "test"):
        path = directory / f"{name}.json"
        if not path.is_file():
            raise FileNotFoundError(f"official FF++ split file missing: {path}")
        for pair in json.loads(path.read_text()):
            for video_id in pair:
                id_to_split[str(video_id)] = name
    assignment: dict[str, str] = {}
    for stem, *_ in entries:
        splits = {id_to_split[i] for i in parse_ids(stem) if i in id_to_split}
        if len(splits) == 1:
            assignment[stem] = splits.pop()
        elif splits:
            raise ValueError(f"{stem} spans multiple official splits: {sorted(splits)}")
    return assignment


def build(cfg: FfppConfig, probe_frames: Callable[[str | Path], int]) -> list[VideoRecord]:
    """Build the manifest. ``probe_frames(path) -> int`` counts decodable frames."""
    entries = scan(cfg)
    if not entries:
        raise FileNotFoundError(f"no videos found under {cfg.root} for compression '{cfg.compression}'")

    if cfg.max_videos_per_class:
        limited: list[tuple[str, Path, str, int]] = []
        per_method: dict[str, int] = {}
        for entry in entries:
            method = entry[2]
            if per_method.get(method, 0) < cfg.max_videos_per_class:
                per_method[method] = per_method.get(method, 0) + 1
                limited.append(entry)
        entries = limited

    stem_to_group = assign_groups(entries)
    splits_dir = resolve_splits_dir(cfg)
    if splits_dir is not None:
        stem_to_split = split_from_dir(splits_dir, entries)
    else:
        group_to_split = split_groups(set(stem_to_group.values()), cfg.split_fractions, cfg.split_seed)
        stem_to_split = {stem: group_to_split[g] for stem, g in stem_to_group.items()}

    records: list[VideoRecord] = []
    for stem, path, method, label in entries:
        if stem not in stem_to_split:
            continue
        n_frames = probe_frames(path)
        if n_frames <= 0:
            continue
        records.append(
            VideoRecord(
                video_id=f"{method}/{stem}",
                path=str(path),
                label=label,
                n_frames=n_frames,
                split=stem_to_split[stem],
                extra={
                    "method": method,
                    "compression": cfg.compression,
                    "group": stem_to_group[stem],
                },
            )
        )
    return records


def check_no_leakage(records: list[VideoRecord]) -> None:
    """Fail if any source identity appears in more than one split."""
    id_to_splits: dict[str, set[str]] = {}
    for record in records:
        stem = record.video_id.split("/", 1)[1]
        for source_id in parse_ids(stem):
            id_to_splits.setdefault(source_id, set()).add(record.split)
    leaked = {i: s for i, s in id_to_splits.items() if len(s) > 1}
    if leaked:
        sample = dict(list(leaked.items())[:5])
        raise AssertionError(f"{len(leaked)} source IDs span multiple splits, e.g. {sample}")


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
                extra={
                    "method": row["method"],
                    "compression": row["compression"],
                    "group": int(row["group"]),
                },
            )
            for row in csv.DictReader(handle)
        ]


def summarise(records: list[VideoRecord]) -> str:
    """Human-readable split/method counts."""
    lines = ["", f"{'split':<8}{'method':<18}{'label':>6}{'videos':>9}{'frames':>12}"]
    lines.append("-" * 53)
    seen: dict[tuple[str, str, int], list[int]] = {}
    for record in records:
        seen.setdefault((record.split, str(record.extra.get("method", "")), record.label), []).append(record.n_frames)
    for (split, method, label), frames in sorted(seen.items()):
        lines.append(f"{split:<8}{method:<18}{label:>6}{len(frames):>9}{sum(frames):>12,}")
    lines.append("-" * 53)
    for split in ("train", "val", "test"):
        subset = [r for r in records if r.split == split]
        n_real = sum(1 for r in subset if r.label == REAL)
        lines.append(f"{split:<8}{len(subset):>5} videos  ({n_real} real / {len(subset) - n_real} fake)")
    lines.append(f"{'total':<8}{len(records):>5} videos")
    return "\n".join(lines)


def ensure_manifest(
    cfg: FfppConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> Path:
    """Create the FF++ manifest CSV if it does not already exist.

    This is FF++-only. Other datasets must not call it. Rank 0 writes; other
    ranks wait on ``barrier``.
    """
    from mars.data.video import count_frames
    from mars.utils.logging import get_logger

    path = Path(cfg.manifest)
    if path.exists():
        return path
    if is_main:
        logger = get_logger()
        logger.info("no FF++ manifest at %s - scanning %s", path, cfg.root)
        splits_dir = resolve_splits_dir(cfg)
        if splits_dir is not None:
            logger.info("using official FF++ splits at %s", splits_dir)
        else:
            logger.info(
                "using random identity-safe fractions %s seed %s",
                cfg.split_fractions,
                cfg.split_seed,
            )
        with ThreadPoolExecutor(max_workers=16) as pool:
            entries = scan(cfg)
            counts = dict(
                zip(
                    (e[1] for e in entries),
                    pool.map(count_frames, (e[1] for e in entries)),
                )
            )
        records = build(cfg, lambda p: counts.get(Path(p), 0))
        check_no_leakage(records)
        write(records, path)
        logger.info("%s", summarise(records))
    if barrier is not None:
        barrier()
    if not path.exists():
        raise FileNotFoundError(
            f"FF++ manifest missing at {path}. Build it with: "
            "python -m mars.data.ffpp --config configs/datasets/ffpp.yaml"
        )
    return path


def load_records(
    cfg: FfppConfig,
    *,
    is_main: bool = True,
    barrier: Callable[[], None] | None = None,
) -> list[VideoRecord]:
    """Return manifest rows, building the CSV on first use."""
    return read(ensure_manifest(cfg, is_main=is_main, barrier=barrier))
