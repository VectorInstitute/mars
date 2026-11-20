"""Evidence: what a model relied on, measured from its outputs, and probes that test it.

Everything here describes what a model relies on, never where a forgery is.
"""

import math
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from marscope.video.prechecks import GRID, MIN_FACE_PATCHES, Face, face_width_patches


TOP_SHARE = 0.05  # share of attribution cells counted as "top"
Scorer = Callable[[np.ndarray, int], float]  # (frames (T, 3, S, S), n_valid) -> fake_prob


# --- measures


def residual_share(frame_contrib: list[np.ndarray], patch_contrib: list[np.ndarray]) -> float | None:
    """Share of MARS's fake evidence carried by the residual (low-level) path.

    Per clip with a positive total: sum of ``patch_contrib`` over its real frames
    divided by the sum of ``frame_contrib``; then the mean over those clips.

    Parameters
    ----------
    frame_contrib : list of numpy.ndarray
        Per clip, (n_valid,) frame contributions.
    patch_contrib : list of numpy.ndarray
        Per clip, (n_valid, 24, 24) patch contributions.

    Returns
    -------
    float or None
        The share, or None if no clip has positive fake evidence.

    Examples
    --------
    >>> residual_share([np.array([1.0, 1.0])], [np.full((2, 24, 24), 0.75 / 576)])
    0.75
    """
    shares = []
    for frames, patches in zip(frame_contrib, patch_contrib, strict=True):
        total = float(frames.sum())
        if total > 0:
            shares.append(float(patches.sum()) / total)
    return round(float(np.mean(shares)), 4) if shares else None


def top_cells(patch_map: np.ndarray, share: float = TOP_SHARE) -> np.ndarray:
    """Boolean mask of the strongest cells that push toward fake.

    Parameters
    ----------
    patch_map : numpy.ndarray
        (24, 24) signed contributions for one frame.
    share : float
        Share of all cells to keep.

    Returns
    -------
    numpy.ndarray
        (24, 24) bool; only positive cells can be selected.
    """
    k = math.ceil(share * patch_map.size)
    flat = patch_map.ravel()
    order = np.argsort(flat)[::-1][:k]
    keep = order[flat[order] > 0]
    mask = np.zeros(flat.shape, dtype=bool)
    mask[keep] = True
    return mask.reshape(patch_map.shape)


def outline_band(face: Face, width: int, height: int, grid: int = GRID) -> np.ndarray:
    """Cells within one patch of the face outline (an ellipse fitted to the face box).

    The frame is squeezed to a square without cropping, so x and y are scaled
    separately.

    Parameters
    ----------
    face : Face
        Detected face, in pixels.
    width, height : int
        Original frame size.
    grid : int
        Grid size (24).

    Returns
    -------
    numpy.ndarray
        (grid, grid) bool.
    """
    cx = (face.x + face.width / 2) / width * grid
    cy = (face.y + face.height / 2) / height * grid
    a = max(face.width / 2 / width * grid, 1e-6)
    b = max(face.height / 2 / height * grid, 1e-6)
    yy, xx = np.mgrid[0:grid, 0:grid] + 0.5
    r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
    return np.abs(r - 1.0) * min(a, b) <= 1.0


def outline_share(
    patch_maps: dict[int, np.ndarray],
    faces: dict[int, list[Face]],
    width: int,
    height: int,
) -> tuple[float | None, int]:
    """Share of MARS's top fake-pushing cells on the face outline, per frame, averaged.

    Only frames whose largest face is at least 5 patches wide count; faces move, so each
    frame uses its own face.

    Parameters
    ----------
    patch_maps : dict of int to numpy.ndarray
        Kept frame index to its (24, 24) ``patch_contrib``.
    faces : dict of int to list of Face
        Kept frame index to detections, largest first.
    width, height : int
        Original frame size.

    Returns
    -------
    tuple of (float or None, int)
        The mean share (None without a usable frame) and the number of frames used.
    """
    shares = []
    for frame, patch_map in patch_maps.items():
        found = faces.get(frame) or []
        if not found or face_width_patches(found[0], width) < MIN_FACE_PATCHES:
            continue
        top = top_cells(patch_map)
        if not top.any():
            continue
        band = outline_band(found[0], width, height)
        shares.append(float((top & band).sum()) / float(top.sum()))
    return (round(float(np.mean(shares)), 4) if shares else None), len(shares)


# --- assumption tests (any model: change the input, re-score)


def face_masks(
    frame_indices: list[int],
    faces: dict[int, list[Face]],
    width: int,
    height: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per clip frame: the face-outline band and as many cells inside the face.

    Only frames whose largest face is at least 5 patches wide get masks. Inside cells
    are the ones nearest the face centre, so the two masks cover the same area.

    Parameters
    ----------
    frame_indices : list of int
        Original index of each real frame in the clip.
    faces : dict of int to list of Face
        Detections per kept frame, largest first.
    width, height : int
        Original frame size.

    Returns
    -------
    tuple of (dict, dict)
        Frame position to (24, 24) bool masks: outline band, and inside.
    """
    outline: dict[int, np.ndarray] = {}
    inside: dict[int, np.ndarray] = {}
    for t, frame in enumerate(frame_indices):
        found = faces.get(frame) or []
        if not found or face_width_patches(found[0], width) < MIN_FACE_PATCHES:
            continue
        face = found[0]
        band = outline_band(face, width, height)
        cx = (face.x + face.width / 2) / width * GRID
        cy = (face.y + face.height / 2) / height * GRID
        a = max(face.width / 2 / width * GRID, 1e-6)
        b = max(face.height / 2 / height * GRID, 1e-6)
        yy, xx = np.mgrid[0:GRID, 0:GRID] + 0.5
        r = np.sqrt(((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2)
        interior = (r < 1.0) & ~band
        n = int(min(band.sum(), interior.sum()))
        if n == 0:
            continue
        inner = np.zeros_like(band)
        order = np.argsort(np.where(interior, r, np.inf), axis=None)[:n]
        inner.flat[order] = True
        outer = band.copy()
        if band.sum() > n:  # keep an evenly spread subset of the band
            cells = np.flatnonzero(band)
            outer[:] = False
            outer.flat[cells[np.linspace(0, len(cells) - 1, n).round().astype(int)]] = True
        outline[t], inside[t] = outer, inner
    return outline, inside


def outline_test(  # noqa: PLR0917 - mirrors the other probes
    score: Scorer,
    frames: np.ndarray,
    n_valid: int,
    frame_indices: list[int],
    faces: dict[int, list[Face]],
    width: int,
    height: int,
) -> dict[str, float] | None:
    """Test A1 for any model: blank the face outline, then as much face interior.

    Parameters
    ----------
    score : callable
        Scores one clip.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    frame_indices : list of int
        Original index of each real frame.
    faces : dict of int to list of Face
        Detections per kept frame.
    width, height : int
        Original frame size.

    Returns
    -------
    dict or None
        ``before``, ``outline_drop``, ``inside_drop`` and ``gap`` (outline minus
        inside), or None when under half of the frames have a usable face.
    """
    outline, inside = face_masks(frame_indices[:n_valid], faces, width, height)
    if len(outline) < n_valid / 2:
        return None
    before = score(frames, n_valid)
    outline_drop = before - score(blank_cells(frames, outline), n_valid)
    inside_drop = before - score(blank_cells(frames, inside), n_valid)
    return {
        "before": before,
        "outline_drop": outline_drop,
        "inside_drop": inside_drop,
        "gap": outline_drop - inside_drop,
    }


def remove_fine_detail(frames: np.ndarray, n_valid: int) -> np.ndarray:
    """Halve each frame and restore its size: fine pixel traces go, what it shows stays.

    Parameters
    ----------
    frames : numpy.ndarray
        (T, 3, S, S) clip in [-1, 1].
    n_valid : int
        Real frames (padding is copied unchanged).

    Returns
    -------
    numpy.ndarray
        A copy with fine detail removed from the real frames.
    """
    out = frames.copy()
    size = frames.shape[-1]
    half = max(1, size // 2)
    for t in range(n_valid):
        for c in range(frames.shape[1]):
            img = Image.fromarray(frames[t, c].astype(np.float32))
            small = img.resize((half, half), Image.Resampling.BICUBIC)
            out[t, c] = np.asarray(small.resize((size, size), Image.Resampling.BICUBIC))
    return np.clip(out, -1.0, 1.0)


def low_level_test(score: Scorer, frames: np.ndarray, n_valid: int) -> dict[str, float]:
    """A5 for any model: the score before and after fine pixel detail is removed.

    Parameters
    ----------
    score : callable
        Scores one clip.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.

    Returns
    -------
    dict
        ``before``, ``after`` and ``share`` (the part of the score lost, 0 to 1).
    """
    before = score(frames, n_valid)
    after = score(remove_fine_detail(frames, n_valid), n_valid)
    share = (before - after) / before if before > 0 else 0.0
    return {"before": before, "after": after, "share": float(np.clip(share, 0.0, 1.0))}


# --- probes


@dataclass(frozen=True)
class ProbeResult:
    """A faithfulness probe on one clip, compared with random removal."""

    probe: str
    count: int
    before: float
    after: float
    random_after: float
    cells: list[list[int]] | None = None  # per real frame, flat indices of the blanked cells

    def as_dict(self) -> dict[str, Any]:
        """Return the result as plain values.

        Returns
        -------
        dict
            Probe name, count and the three scores, rounded to three decimals, plus
            the blanked cells of each frame when the probe hides areas.
        """
        out: dict[str, Any] = {
            "probe": self.probe,
            "count": self.count,
            "before": round(self.before, 3),
            "after": round(self.after, 3),
            "random_after": round(self.random_after, 3),
        }
        if self.cells is not None:
            out["cells"] = self.cells
        return out


def remove_frames(frames: np.ndarray, n_valid: int, drop: set[int]) -> tuple[np.ndarray, int]:
    """Remove real frames from a clip, then pad by repeating the last remaining frame.

    Parameters
    ----------
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    drop : set of int
        Frame positions to remove.

    Returns
    -------
    tuple of (numpy.ndarray, int)
        The new clip and its ``n_valid`` (at least 1).
    """
    keep = [t for t in range(n_valid) if t not in drop] or [0]
    kept = frames[keep]
    pad = np.repeat(kept[-1:], frames.shape[0] - len(keep), axis=0)
    return np.concatenate([kept, pad]), len(keep)


def segments(n_valid: int, count: int = 16) -> list[list[int]]:
    """Split a clip's real frames into equal runs, one per frame shown in the UI.

    Parameters
    ----------
    n_valid : int
        Real frames in the clip.
    count : int
        Runs wanted (fewer when the clip is shorter).

    Returns
    -------
    list of list of int
        Frame positions of each run, in order.

    Examples
    --------
    >>> segments(8, 4)
    [[0, 1], [2, 3], [4, 5], [6, 7]]
    >>> segments(3, 16)
    [[0], [1], [2]]
    """
    k = max(1, min(count, n_valid))
    return [list(range(i * n_valid // k, (i + 1) * n_valid // k)) for i in range(k)]


def drop_segments(segs: list[list[int]], picked: list[int]) -> set[int]:
    """Frame positions covered by the picked runs.

    Parameters
    ----------
    segs : list of list of int
        Runs from :func:`segments`.
    picked : list of int
        Run numbers to remove.

    Returns
    -------
    set of int
        Frame positions to remove.

    Examples
    --------
    >>> sorted(drop_segments([[0, 1], [2, 3], [4, 5]], [0, 2]))
    [0, 1, 4, 5]
    """
    return {t for i in picked if 0 <= i < len(segs) for t in segs[i]}


def sweep_frames(  # noqa: PLR0913, PLR0917
    score: Scorer,
    frames: np.ndarray,
    n_valid: int,
    values: np.ndarray,
    count: int = 16,
    max_k: int = 8,
    draws: int = 3,
    seed: int = 0,
) -> dict[str, list[float] | list[int]]:
    """Score a clip as more of its top runs are removed, against random runs.

    Parameters
    ----------
    score : callable
        Scores one clip.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    values : numpy.ndarray
        Per real frame, higher means more toward fake (the model's own values).
    count : int
        Runs the clip is split into (one per frame shown).
    max_k : int
        Largest number of runs removed.
    draws : int
        Random draws averaged at each step.
    seed : int
        Random seed (fixed, so the same clip gives the same curve).

    Returns
    -------
    dict
        ``order`` (runs ranked by the model's values), ``top`` and ``random`` (score
        after removing 0..max_k runs).
    """
    segs = segments(n_valid, count)
    means = [float(np.mean(values[s])) for s in segs]
    order = [int(i) for i in np.argsort(means)[::-1]]
    max_k = min(max_k, len(segs) - 1)
    before = score(frames, n_valid)
    top = [before] + [
        score(*remove_frames(frames, n_valid, drop_segments(segs, order[:k]))) for k in range(1, max_k + 1)
    ]
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(len(segs)).tolist() for _ in range(draws)]
    rand = [before] + [
        float(np.mean([score(*remove_frames(frames, n_valid, drop_segments(segs, p[:k]))) for p in perms]))
        for k in range(1, max_k + 1)
    ]
    return {"order": order, "top": top, "random": rand}


def blank_cells(frames: np.ndarray, masks: dict[int, np.ndarray], grid: int = GRID) -> np.ndarray:
    """Set grid cells to mid-grey (0 in [-1, 1]) in the given frames.

    Parameters
    ----------
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    masks : dict of int to numpy.ndarray
        Frame position to a (grid, grid) bool mask of cells to blank.
    grid : int
        Grid size.

    Returns
    -------
    numpy.ndarray
        A copy with the cells blanked.
    """
    out = frames.copy()
    size = frames.shape[-1]
    for t, mask in masks.items():
        for r, c in zip(*np.nonzero(mask), strict=True):
            y0, y1 = r * size // grid, (r + 1) * size // grid
            x0, x1 = c * size // grid, (c + 1) * size // grid
            out[t, :, y0:y1, x0:x1] = 0.0
    return out


def probe_frames(score: Scorer, frames: np.ndarray, n_valid: int, values: np.ndarray, seed: int = 0) -> ProbeResult:
    """Remove the frames the model relied on most; compare with removing random frames.

    Parameters
    ----------
    score : callable
        Scores one clip: ``score(frames, n_valid) -> fake_prob``.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    values : numpy.ndarray
        Per real frame: MARS contribution, or frame probability; higher means more
        toward fake.
    seed : int
        Random seed (fixed, so the same clip gives the same result).

    Returns
    -------
    ProbeResult
        Scores before, after removing the top frames, and the mean over three random
        removals.
    """
    k = max(1, round(0.1 * n_valid))
    before = score(frames, n_valid)
    top = set(np.argsort(values[:n_valid])[::-1][:k].tolist())
    after = score(*remove_frames(frames, n_valid, top))
    rng = np.random.default_rng(seed)
    randoms = []
    for _ in range(3):
        pick = set(rng.choice(n_valid, size=k, replace=False).tolist())
        randoms.append(score(*remove_frames(frames, n_valid, pick)))
    return ProbeResult("remove_top_frames", k, before, after, float(np.mean(randoms)))


def probe_patches(
    score: Scorer,
    frames: np.ndarray,
    n_valid: int,
    patch_contrib: np.ndarray,
    seed: int = 0,
) -> ProbeResult:
    """Blank each frame's top fake-pushing cells (MARS); compare with random cells.

    Parameters
    ----------
    score : callable
        Scores one clip.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    patch_contrib : numpy.ndarray
        (n_valid, 24, 24) MARS patch contributions.
    seed : int
        Random seed.

    Returns
    -------
    ProbeResult
        Scores before, after blanking, and the mean over three random blankings, with
        the blanked cells of each frame (flat indices into the 24 x 24 grid).
    """
    before = score(frames, n_valid)
    masks = {t: top_cells(patch_contrib[t]) for t in range(n_valid)}
    after = score(blank_cells(frames, masks), n_valid)
    rng = np.random.default_rng(seed)
    randoms = []
    for _ in range(3):
        random_masks = {}
        for t, mask in masks.items():
            m = np.zeros(mask.size, dtype=bool)
            m[rng.choice(mask.size, size=int(mask.sum()), replace=False)] = True
            random_masks[t] = m.reshape(mask.shape)
        randoms.append(score(blank_cells(frames, random_masks), n_valid))
    count = int(sum(int(m.sum()) for m in masks.values()))
    cells = [np.flatnonzero(masks[t]).tolist() for t in range(n_valid)]
    return ProbeResult("blank_top_patches", count, before, after, float(np.mean(randoms)), cells)


AREA_CELLS = 8  # cells of the 8 x 8 occlusion grid hidden together by the areas test (1/8 of the frame)


def probe_occlusion(
    score: Scorer,
    frames: np.ndarray,
    n_valid: int,
    occlusion: np.ndarray,
    *,
    k: int = AREA_CELLS,
    seed: int = 0,
) -> ProbeResult:
    """Hide a model's top occlusion cells in every frame; compare with random cells.

    The same test as :func:`probe_patches`, for models without a built-in map (GenD,
    DFD-HR): the cells are the ``k`` that lowered the score most when hidden one by one.

    Parameters
    ----------
    score : callable
        Scores one clip.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    occlusion : numpy.ndarray
        (grid, grid) occlusion map (drop in the score per hidden cell).
    k : int
        Cells hidden.
    seed : int
        Random seed.

    Returns
    -------
    ProbeResult
        Scores before, after, and the mean over three random draws; ``cells`` gives the
        hidden cells of each frame on the 24 x 24 display grid.

    Examples
    --------
    >>> occ = np.zeros((8, 8))
    >>> occ[0, 0] = 1.0
    >>> clip = np.ones((2, 3, 24, 24))
    >>> res = probe_occlusion(lambda f, n: float(f.mean()), clip, 2, occ, k=1)
    >>> res.count, res.cells[0][:3]
    (1, [0, 1, 2])
    """
    grid = occlusion.shape[0]
    flat = np.asarray(occlusion, dtype=np.float64).ravel()
    top = np.argsort(-flat, kind="stable")[:k]
    chosen = np.zeros(flat.size, dtype=bool)
    chosen[top] = True
    mask = chosen.reshape(grid, grid)
    before = score(frames, n_valid)
    after = score(blank_cells(frames, dict.fromkeys(range(n_valid), mask), grid=grid), n_valid)
    rng = np.random.default_rng(seed)
    randoms = []
    for _ in range(3):
        m = np.zeros(flat.size, dtype=bool)
        m[rng.choice(flat.size, size=k, replace=False)] = True
        randoms.append(
            score(blank_cells(frames, dict.fromkeys(range(n_valid), m.reshape(grid, grid)), grid=grid), n_valid)
        )
    scale = GRID // grid  # each occlusion cell covers scale x scale display cells
    cells = [divmod(int(i), grid) for i in top]
    display = sorted(
        (r * scale + dy) * GRID + c * scale + dx for r, c in cells for dy in range(scale) for dx in range(scale)
    )
    return ProbeResult("blank_top_areas", k, before, after, float(np.mean(randoms)), [display] * n_valid)


def occlusion_map(
    score: Scorer,
    frames: np.ndarray,
    n_valid: int,
    grid: int = 8,
    guard: AbstractContextManager[Any] | None = None,
) -> np.ndarray:
    """Model-agnostic attribution: blank one grid cell in every frame and re-score.

    Parameters
    ----------
    score : callable
        Scores one clip.
    frames : numpy.ndarray
        (T, 3, S, S) clip.
    n_valid : int
        Real frames.
    grid : int
        Cells per side (8: 64 passes).
    guard : context manager, optional
        Held around each pass only (for example the server's GPU lock), so other work
        can use the GPU between passes instead of waiting for the whole map.

    Returns
    -------
    numpy.ndarray
        (grid, grid) drop in ``fake_prob`` when the cell is blanked; positive means the
        cell pushed toward fake.
    """
    hold = guard if guard is not None else nullcontext()
    with hold:
        base = score(frames, n_valid)
    result = np.zeros((grid, grid), dtype=np.float32)
    for r in range(grid):
        for c in range(grid):
            mask = np.zeros((grid, grid), dtype=bool)
            mask[r, c] = True
            blanked = blank_cells(frames, dict.fromkeys(range(n_valid), mask), grid=grid)
            with hold:
                result[r, c] = base - score(blanked, n_valid)
    return result
