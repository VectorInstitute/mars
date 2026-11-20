"""Tests for usual ranges, evidence measures and probes."""

import numpy as np
import pytest

from marscope.analysis import evidence as ev
from marscope.analysis.ranges import UsualRanges, rank
from marscope.video.prechecks import Face


def ranges() -> UsualRanges:
    return UsualRanges.from_dict(
        {
            "real": np.linspace(0.0, 0.5, 101).tolist(),
            "fake": np.linspace(0.2, 1.0, 101).tolist(),
            "n_real": 10,
            "n_fake": 10,
        }
    )


def test_rank_interpolates_and_clips() -> None:
    p = np.linspace(0.0, 1.0, 101)
    assert rank(0.505, p) == pytest.approx(50.5)
    assert rank(-1.0, p) == 0.0
    assert rank(1.0, p) == 100.0


def test_rank_flat_stretch() -> None:
    p = np.array([0.0] * 50 + [0.5] * 51)
    assert rank(0.5, p) == 100.0
    assert rank(0.25, p) == pytest.approx(49.5)


def test_context_high_fake_score_is_above_real_p95() -> None:
    r = ranges()
    assert r.real_range() == pytest.approx((0.025, 0.475))
    assert not r.context(0.475).high_fake_score
    assert r.context(0.476).high_fake_score
    assert r.context(0.3).real_rank == pytest.approx(60.0)


def test_ranges_reject_bad_files() -> None:
    with pytest.raises(ValueError):
        UsualRanges.from_dict({"real": [0.0] * 100, "fake": [0.0] * 101})
    with pytest.raises(ValueError):
        UsualRanges.from_dict({"real": list(range(101))[::-1], "fake": [0.0] * 101})


def test_residual_share_skips_negative_clips() -> None:
    fc = [np.array([1.0, 1.0]), np.array([-1.0, -1.0])]
    pc = [np.full((2, 24, 24), 0.5 / 576), np.zeros((2, 24, 24))]
    assert ev.residual_share(fc, pc) == pytest.approx(0.5)
    assert ev.residual_share([np.array([-1.0])], [np.zeros((1, 24, 24))]) is None


def test_top_cells_only_positive() -> None:
    m = np.full((24, 24), -1.0)
    m[3, 4] = 2.0
    top = ev.top_cells(m)
    assert top.sum() == 1
    assert top[3, 4]


def test_outline_share_per_frame_and_small_faces() -> None:
    face = Face(x=480, y=180, width=320, height=360, score=0.9, keypoints=())  # 6 patches wide at 1280 px
    band = ev.outline_band(face, 1280, 720)
    on_band = np.where(band, 1.0, -1.0)
    inside = np.full((24, 24), -1.0)
    inside[12, 12] = 1.0  # face centre
    share, used = ev.outline_share({0: on_band, 2: inside}, {0: [face], 2: [face]}, 1280, 720)
    assert used == 2
    assert share == pytest.approx(0.5)
    tiny = Face(x=0, y=0, width=100, height=100, score=0.9, keypoints=())
    assert ev.outline_share({0: on_band}, {0: [tiny]}, 1280, 720) == (None, 0)


def test_remove_frames_pads_with_last_kept() -> None:
    frames = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1) * np.ones((4, 3, 2, 2), dtype=np.float32)
    out, n = ev.remove_frames(frames, 3, {1})
    assert n == 2
    assert [float(out[t, 0, 0, 0]) for t in range(4)] == [0.0, 2.0, 2.0, 2.0]


def test_probes_and_occlusion_with_a_fake_scorer() -> None:
    frames = np.zeros((8, 3, 24, 24), dtype=np.float32)
    frames[:, :, :12, :12] = 1.0  # the "evidence" is in the top-left quarter

    def score(f: np.ndarray, n_valid: int) -> float:
        return float(f[:n_valid, :, :12, :12].mean())

    values = np.arange(8, dtype=np.float32)
    result = ev.probe_frames(score, frames, 8, values)
    assert result.count == 1
    assert result.before == pytest.approx(1.0)
    patch = np.zeros((8, 24, 24), dtype=np.float32)
    patch[:, :12, :12] = 1.0
    blanked = ev.probe_patches(score, frames, 8, patch)
    assert blanked.after < blanked.before
    assert blanked.cells is not None and len(blanked.cells) == 8
    assert blanked.count == sum(len(c) for c in blanked.cells)
    assert blanked.as_dict()["cells"] == blanked.cells
    occ = ev.occlusion_map(score, frames, 8, grid=2)
    assert occ[0, 0] > 0
    assert occ[1, 1] == pytest.approx(0.0)


def test_face_masks_cover_equal_area() -> None:
    face = Face(x=480, y=180, width=320, height=360, score=0.9, keypoints=())
    outline, inside = ev.face_masks([0, 2], {0: [face]}, 1280, 720)
    assert list(outline) == [0]
    assert outline[0].sum() == inside[0].sum() > 0
    assert not (outline[0] & inside[0]).any()


def test_outline_and_low_level_tests_with_a_fake_scorer() -> None:
    face = Face(x=480, y=180, width=320, height=360, score=0.9, keypoints=())
    band = ev.outline_band(face, 1280, 720)
    frames = np.zeros((4, 3, 24, 24), dtype=np.float32)
    frames[:, :, band] = 1.0  # the score lives on the outline

    def score(f: np.ndarray, n_valid: int) -> float:
        return float(f[:n_valid, :, band].mean())

    faces = {i: [face] for i in (0, 2, 4, 6)}
    test = ev.outline_test(score, frames, 4, [0, 2, 4, 6], faces, 1280, 720)
    assert test is not None
    assert test["outline_drop"] > 0
    assert test["inside_drop"] == pytest.approx(0.0)
    assert ev.outline_test(score, frames, 4, [0, 2, 4, 6], {}, 1280, 720) is None

    checker = np.indices((24, 24)).sum(axis=0) % 2 * 2.0 - 1.0  # fine detail only
    fine = np.broadcast_to(checker, (4, 3, 24, 24)).astype(np.float32).copy()

    def detail(f: np.ndarray, n_valid: int) -> float:
        return float(np.abs(np.diff(f[:n_valid], axis=-1)).mean())

    low = ev.low_level_test(detail, fine, 4)
    assert low["share"] > 0.5


def test_sweep_ranks_runs_by_the_models_values() -> None:
    frames = np.zeros((8, 3, 4, 4), dtype=np.float32)
    values = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.float32)

    def score(clip: np.ndarray, n_valid: int) -> float:  # fake evidence lives in the last two frames
        return 0.9 if n_valid == 8 else 0.2 + 0.0 * float(clip.sum())

    out = ev.sweep_frames(score, frames, 8, values, count=4, max_k=2, draws=2)
    assert out["order"][0] == 3
    assert len(out["top"]) == 3 and out["top"][0] == 0.9


def test_segments_cover_every_frame_once() -> None:
    segs = ev.segments(61, 16)
    assert len(segs) == 16 and sorted(t for s in segs for t in s) == list(range(61))


def test_clip_event_carries_live_score() -> None:
    from types import SimpleNamespace  # noqa: PLC0415

    from marscope.pipeline import _clip_event  # noqa: PLC0415
    from marscope.schema import ClipScore  # noqa: PLC0415

    clip = ClipScore(index=2, first_frame=256, last_frame=382, n_valid=64, fake=0.91234, frame_values=[0.0])
    runner = SimpleNamespace(ranges=ranges())
    out = _clip_event("mars", clip, runner, 3, 5)  # type: ignore[arg-type]
    assert out["type"] == "clip" and out["index"] == 2 and out["done"] == 3 and out["total"] == 5
    assert out["fake"] == 0.9123
    assert out["real_rank"] == round(rank(0.91234, runner.ranges.real), 1)
    assert out["real_high"] == round(runner.ranges.real_range()[1], 4)


def test_areas_test_hides_the_top_occlusion_cells() -> None:
    frames = np.ones((4, 3, 24, 24), dtype=np.float32)
    occ = np.zeros((8, 8))
    occ[2, 5] = 0.5
    occ[7, 0] = 0.4

    def score(f: np.ndarray, n: int) -> float:
        return float(f[:n].mean())

    res = ev.probe_occlusion(score, frames, 4, occ, k=2, seed=1)
    assert res.count == 2 and res.after < res.before
    assert len(res.cells) == 4 and len(res.cells[0]) == 18  # two cells, each 3 x 3 on the display grid
    assert 2 * 3 * 24 + 5 * 3 in res.cells[0] and 7 * 3 * 24 in res.cells[0]
