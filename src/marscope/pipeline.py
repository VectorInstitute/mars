"""Pre-checks, decode, run the models, place scores in context, apply the policy."""

import json
import logging
import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from marscope.analysis import evidence as ev
from marscope.analysis.ranges import REAL_LEAN_RANK, rank
from marscope.explain.policy import apply_policy, assess, load_policy, policy_version
from marscope.models.registry import get_spec, parse_models
from marscope.models.runner import Runner, cuda_available, download, peak_vram_gb
from marscope.models.scheduler import free_vram_gb, plan_waves
from marscope.schema import (
    Agreement,
    ClipScore,
    ModelScore,
    Prechecks,
    Precision,
    Versions,
    VideoInfo,
    VideoResult,
    to_float,
)
from marscope.video.edits import SHARE_STEPS, Edits, apply_edits
from marscope.video.prechecks import (
    Face,
    FaceFinder,
    bits_per_pixel,
    read_credentials,
    scope_warnings,
    summarize_faces,
)
from marscope.video.preprocess import Clip, build_clips, iter_kept_frames, read_clip, read_clips, read_meta


log = logging.getLogger(__name__)

Progress = Callable[[dict[str, Any]], None]
THUMB_HEIGHT = 360


def upright(img: Image.Image, rotation: int) -> Image.Image:
    """Rotate a thumbnail upright (the models still see the frame as decoded).

    Parameters
    ----------
    img : PIL.Image.Image
        A decoded frame.
    rotation : int
        Degrees counter-clockwise, from the file's display rotation.

    Returns
    -------
    PIL.Image.Image
        The rotated image (the same image when the rotation is 0).

    Examples
    --------
    >>> upright(Image.new("RGB", (4, 2)), 90).size
    (2, 4)
    """
    return img.rotate(rotation, expand=True) if rotation % 360 else img


TEST_CLIPS = 3  # strongest clips used by the A1 and A5 input tests
SHOWN_FRAMES = 16  # runs a clip is split into in the UI (one shown frame per run)
QUALITIES = (100, 70, 50, 30, 10)  # JPEG qualities of the what-if chart


def _editor(edits: Edits) -> Callable[[int, np.ndarray], np.ndarray]:
    def edit(index: int, rgb: np.ndarray) -> np.ndarray:
        return apply_edits(rgb, edits, seed=index)

    return edit


def package_version() -> str:
    """Return the installed MARScope version.

    Returns
    -------
    str
        The version, or ``0+unknown`` when running from a source tree that is not
        installed.
    """
    try:
        return version("marscope")
    except PackageNotFoundError:
        return "0+unknown"


@dataclass
class _ModelRun:
    clips: list[ClipScore] = field(default_factory=list)
    frame_contrib: list[np.ndarray] = field(default_factory=list)
    patch_contrib: list[np.ndarray] = field(default_factory=list)
    gate: list[np.ndarray] = field(default_factory=list)
    tests: dict[str, float] = field(default_factory=dict)
    seconds: float = 0.0


def _clip_event(model_id: str, clip: ClipScore, runner: Runner, done: int, total: int) -> dict[str, Any]:
    """Progress event for one scored clip, so the UI can show results as they arrive.

    Parameters
    ----------
    model_id : str
        Model that scored the clip.
    clip : ClipScore
        The clip's score.
    runner : Runner
        The model's runner (for its usual real range).
    done, total : int
        Clips scored so far, and the expected count.

    Returns
    -------
    dict
        ``type`` "clip", the model, counts, the clip's frames and score, and the top of
        the model's benchmark real range.
    """
    return {
        "type": "clip",
        "model": model_id,
        "done": done,
        "total": total,
        "index": clip.index,
        "first_frame": clip.first_frame,
        "last_frame": clip.last_frame,
        "fake": round(clip.fake, 4),
        "real_rank": round(rank(clip.fake, runner.ranges.real), 1),
        "real_high": round(runner.ranges.real_range()[1], 4),
    }


def evidence_strength(model_ids: list[str], high: list[str], scores: dict[str, ModelScore]) -> Agreement:
    """Summarize how consistent the models are, by their usual ranges. Never a verdict.

    Mixed when the models disagree; consistent when all give a high fake score, or when
    none does and every score sits in the lower part of its benchmark real range; weak
    otherwise.

    Parameters
    ----------
    model_ids : list of str
        Models that ran.
    high : list of str
        Models with a high fake score.
    scores : dict of str to ModelScore
        Their scores.

    Returns
    -------
    Agreement
        Agreement, strength and lean.
    """
    disagree = len(model_ids) > 1 and 0 < len(high) < len(model_ids)
    if disagree:
        return Agreement(models=model_ids, high_fake=high, disagree=True, strength="mixed")
    if len(high) == len(model_ids):
        return Agreement(models=model_ids, high_fake=high, disagree=False, strength="consistent", lean="fake")
    if all(scores[m].context.real_rank < REAL_LEAN_RANK for m in model_ids):
        return Agreement(models=model_ids, high_fake=high, disagree=False, strength="consistent", lean="real")
    return Agreement(models=model_ids, high_fake=high, disagree=False, strength="weak")


class MARScope:
    """Run one or more detectors on a video and explain the result.

    Parameters
    ----------
    models : str or list of str, optional
        ``"mars,gend"``, a list, ``"all"`` for all three, or None for MARS only.
    device : {"auto", "cuda", "cpu"}
        Where to run the models.
    sequential : bool
        Run one model at a time, whatever the free GPU memory.
    faces : bool
        Detect faces (needed by A1 and the face checks of A3).
    device_id : int
        GPU index.
    precision : {"auto", "fp16", "fp32"}
        ``auto`` is fp16 on the GPU and fp32 on the CPU.

    Examples
    --------
    >>> scope = MARScope(models="mars")  # doctest: +SKIP
    >>> result = scope.analyze("clip.mp4", jurisdiction="ca")  # doctest: +SKIP
    >>> result.models["mars"].fake  # doctest: +SKIP
    """

    def __init__(  # noqa: PLR0917
        self,
        models: str | list[str] | None = None,
        device: str = "auto",
        sequential: bool = False,
        faces: bool = True,
        device_id: int = 0,
        precision: str = "auto",
    ) -> None:
        if device not in ("auto", "cuda", "cpu"):
            raise ValueError("device must be auto, cuda or cpu")
        self.model_ids = parse_models(models)
        self.device = device
        self.sequential = sequential
        self.faces = faces
        self.device_id = device_id
        if precision not in ("auto", "fp16", "fp32"):
            raise ValueError("precision must be auto, fp16 or fp32")
        self.precision = precision
        self._runners: dict[str, Runner] = {}

    # ---
    # model loading

    def load(self, model_id: str) -> Runner:
        """Load a model (downloading it on first use) and keep it loaded.

        Parameters
        ----------
        model_id : str
            Model id.

        Returns
        -------
        Runner
            The loaded model.
        """
        if model_id not in self._runners:
            self._runners[model_id] = Runner(
                get_spec(model_id), device=self.device, device_id=self.device_id, precision=self.precision
            )
        return self._runners[model_id]

    def unload(self, model_id: str) -> None:
        """Release a loaded model.

        Parameters
        ----------
        model_id : str
            Model id.
        """
        self._runners.pop(model_id, None)

    def loaded(self) -> list[str]:
        """Return the ids of the loaded models.

        Returns
        -------
        list of str
            Loaded model ids.
        """
        return list(self._runners)

    def plan(self, model_ids: list[str]) -> list[list[str]]:
        """Plan waves of models that fit in free GPU memory.

        Parameters
        ----------
        model_ids : list of str
            Models to run.

        Returns
        -------
        list of list of str
            Waves, as from :func:`marscope.models.scheduler.plan_waves`.
        """
        on_gpu = self.device == "cuda" or (self.device == "auto" and cuda_available())
        if not on_gpu or len(model_ids) == 1:
            return [[m] for m in model_ids] if not on_gpu else [model_ids]
        peaks = {}
        for m in model_ids:
            serving = json.loads(download(get_spec(m), "serving.json").read_text())
            precision: Precision = "fp32" if self.precision == "fp32" else "fp16"
            peaks[m] = 0.0 if m in self._runners else peak_vram_gb(serving, precision)
        return plan_waves(peaks, free_vram_gb(self.device_id), sequential=self.sequential)

    # --- analysis

    def analyze(  # noqa: PLR0912, PLR0915 - one linear pass, easier to read whole
        self,
        path: str | Path,
        jurisdiction: str | None = None,
        *,
        language: str = "en",
        models: str | list[str] | None = None,
        progress: Progress | None = None,
        frames_dir: str | Path | None = None,
    ) -> VideoResult:
        """Analyze one video.

        Parameters
        ----------
        path : str or Path
            Video file.
        jurisdiction : {"eu", "ca", "us"} or None
            Region for context notes.
        language : {"en", "fr"}
            Note language.
        models : str or list of str, optional
            Override the toolkit's models for this video.
        progress : callable, optional
            Receives events: ``{"type": "stage", ...}``, ``{"type": "clip", "model",
            "done", "total"}``.
        frames_dir : str or Path, optional
            If given, kept frames are saved there as JPEG thumbnails (for the UI).

        Returns
        -------
        VideoResult
            Scores, context, evidence measures and policy notes.
        """
        path = Path(path)
        model_ids = parse_models(models) if models is not None else self.model_ids
        emit = progress or (lambda _event: None)
        meta = read_meta(path)
        emit(
            {"type": "stage", "stage": "prechecks", "fps": round(meta.fps, 3), "duration_s": round(meta.duration_s, 3)}
        )
        credentials = read_credentials(path)

        thumbs = Path(frames_dir) if frames_dir is not None else None
        if thumbs is not None:
            thumbs.mkdir(parents=True, exist_ok=True)
        faces: dict[int, list[Face]] = {}
        finder = self._face_finder() if self.faces else None

        runs: dict[str, _ModelRun] = {m: _ModelRun() for m in model_ids}
        runners: dict[str, Runner] = {}
        n_kept = 0
        last_index = -1
        n_clips = 0
        estimate = max(1, math.ceil(meta.duration_s * meta.fps / 2 / 64)) if meta.fps else 1

        def on_frame(index: int, rgb: np.ndarray) -> None:
            nonlocal n_kept, last_index
            n_kept += 1
            last_index = index
            if finder is not None:
                faces[index] = finder.detect(rgb)
            if thumbs is not None:
                img = upright(Image.fromarray(rgb), meta.rotation)
                img.thumbnail((THUMB_HEIGHT * 4, THUMB_HEIGHT))
                img.save(thumbs / f"{index:06d}.jpg", quality=85)

        try:
            for wave_number, wave in enumerate(self.plan(model_ids)):
                for loaded in list(self._runners):
                    if loaded not in wave and len(model_ids) > len(wave):
                        self.unload(loaded)
                for m in wave:
                    emit({"type": "stage", "stage": "loading", "model": m})
                    runners[m] = self.load(m)
                protocol = runners[wave[0]].protocol
                for m in wave:
                    p = runners[m].protocol
                    if (p.n_frames, p.frame_stride) != (protocol.n_frames, protocol.frame_stride):
                        raise ValueError("selected models use different clip protocols")
                sizes = {runners[m].protocol.image_size for m in wave}
                first = wave_number == 0
                frames = iter_kept_frames(path, protocol.frame_stride)
                clips = build_clips(frames, sizes, protocol.n_frames, on_frame=on_frame if first else None)
                emit({"type": "stage", "stage": "running", "models": wave})
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    for clip in clips:
                        list(pool.map(lambda m, c=clip: self._run_clip(runners[m], c, runs[m]), wave))
                        n_clips = clip.index + 1
                        for m in wave:
                            emit(_clip_event(m, runs[m].clips[-1], runners[m], n_clips, max(estimate, n_clips)))
                for m in wave:
                    self._assumption_tests(path, runners[m], runs[m], faces, (meta.width, meta.height), emit)
        finally:
            if finder is not None:
                finder.close()

        info = VideoInfo(
            name=path.name,
            duration_s=round(meta.duration_s, 3),
            width=meta.width,
            height=meta.height,
            fps=round(meta.fps, 3),
            n_frames=last_index + 1,
            n_kept_frames=n_kept,
            n_clips=n_clips,
            frame_stride=runners[model_ids[0]].protocol.frame_stride,
            rotation=meta.rotation,
        )
        n_per_clip = runners[model_ids[0]].protocol.n_frames
        prechecks = Prechecks(
            in_scope=not scope_warnings(n_kept, n_per_clip, meta.width, meta.height),
            scope_warnings=scope_warnings(n_kept, n_per_clip, meta.width, meta.height),
            bits_per_pixel=bits_per_pixel(meta.size_bytes, last_index + 1, meta.width, meta.height),
            faces=summarize_faces(faces, meta.width) if finder is not None else None,
            credentials=credentials,
        )
        scores = {m: self._score(runners[m], runs[m], faces, meta.width, meta.height) for m in model_ids}
        high = [m for m in model_ids if scores[m].context.high_fake_score]
        agreement = evidence_strength(model_ids, high, scores)
        result = VideoResult(
            video=info,
            prechecks=prechecks,
            models=scores,
            agreement=agreement,
            versions=Versions(
                package=package_version(),
                protocol=f"MARS protocol: {n_per_clip}-frame clips, stride {runners[model_ids[0]].protocol.frame_stride}",
                policy=policy_version(),
                checkpoints={m: runners[m].commit for m in model_ids},
            ),
        )
        emit({"type": "stage", "stage": "policy"})
        result.notes = apply_policy(result, jurisdiction, language)
        result.reading = assess(result, language)
        result.jurisdiction = jurisdiction
        for m in model_ids:
            # Percentiles used (matching the precision), to rank re-scored values.
            arrays: dict[str, Any] = {
                "real_percentiles": runners[m].ranges.real,
                "fake_percentiles": runners[m].ranges.fake,
            }
            if runs[m].patch_contrib:
                arrays |= {"patch_contrib": runs[m].patch_contrib, "gate": runs[m].gate}
            result.set_evidence(m, arrays)
        emit({"type": "stage", "stage": "done"})
        return result

    def _face_finder(self) -> FaceFinder | None:
        try:
            return FaceFinder(min_confidence=float(load_policy()["cutoffs"]["face_confidence"]))
        except Exception as error:  # noqa: BLE001 - faces are optional evidence; report and continue
            log.warning("face detection unavailable (%s); A1 and the face checks of A3 are skipped", error)
            return None

    @staticmethod
    def _run_clip(runner: Runner, clip: Clip, run: _ModelRun) -> None:
        start = time.perf_counter()
        frames = clip.inputs[runner.protocol.image_size][None]
        out = runner.run(frames, np.array([clip.n_valid], dtype=np.int64))
        n = clip.n_valid
        if "frame_contrib" in out:
            values = out["frame_contrib"][0, :n]
            run.frame_contrib.append(values.astype(np.float32))
            run.patch_contrib.append(out["patch_contrib"][0, :n].astype(np.float32))
            run.gate.append(out["gate"][0].astype(np.float32))
        else:
            values = out["frame_probs"][0, :n]
        run.clips.append(
            ClipScore(
                index=clip.index,
                first_frame=clip.frame_indices[0],
                last_frame=clip.frame_indices[-1],
                n_valid=n,
                fake=to_float(out["fake_prob"][0]),
                frame_values=[round(float(v), 4) for v in values],
            ),
        )
        run.seconds += time.perf_counter() - start

    @staticmethod
    def _assumption_tests(  # noqa: PLR0917
        path: Path,
        runner: Runner,
        run: _ModelRun,
        faces: dict[int, list[Face]],
        size: tuple[int, int],
        emit: Progress,
    ) -> None:
        """Run the A1 and A5 input tests on a model's strongest clips (any score).

        Records the mean clip score before and after each change, so the notes can
        answer "does this model break when ..." with numbers.
        """
        emit({"type": "stage", "stage": "testing", "model": runner.spec.id})
        top = [c.index for c in sorted(run.clips, key=lambda c: c.fake, reverse=True)[:TEST_CLIPS]]
        p = runner.protocol
        clips = read_clips(path, top, p.image_size, p.n_frames, p.frame_stride)

        def score(frames: np.ndarray, n_valid: int) -> float:
            return float(runner.run(frames[None], np.array([n_valid], dtype=np.int64))["fake_prob"][0])

        outline: list[dict[str, float]] = []
        low: list[dict[str, float]] = []
        befores: dict[int, float] = {}
        model_id = runner.spec.id

        def step(name: str, index: int, n: int, clip: Clip) -> None:
            """Tell the UI which test runs on which clip, so the wait is visible."""
            mid = clip.frame_indices[len(clip.frame_indices) // 2] if clip.frame_indices else None
            emit({"type": "step", "model": model_id, "test": name, "clip": index, "n": n, "of": len(top), "frame": mid})

        for n, index in enumerate(top, start=1):
            clip = clips.get(index)
            if clip is None:
                continue
            frames = clip.inputs[p.image_size]
            if faces:
                step("a1", index, n, clip)
                test = ev.outline_test(score, frames, clip.n_valid, clip.frame_indices, faces, *size)
                if test is not None:
                    outline.append(test)
            step("a5", index, n, clip)
            low.append(ev.low_level_test(score, frames, clip.n_valid))
            befores[index] = low[-1]["before"]
        # A4: the same clips after "shared many times" (resolution, blur, noise, JPEG).
        a4: list[tuple[float, float]] = []
        if befores:
            shared = _editor(SHARE_STEPS[-1])
            edited = read_clips(path, list(befores), p.image_size, p.n_frames, p.frame_stride, shared)
            for n, (i, c) in enumerate(((i, c) for i, c in edited.items() if i in befores), start=1):
                step("a4", i, n, c)
                a4.append((befores[i], score(c.inputs[p.image_size], c.n_valid)))

        def mean(xs: list[float]) -> float:
            return round(float(np.mean(xs)), 4)

        if outline:
            run.tests["a1_before"] = mean([t["before"] for t in outline])
            run.tests["a1_after"] = mean([t["before"] - t["outline_drop"] for t in outline])
            run.tests["a1_inside"] = mean([t["before"] - t["inside_drop"] for t in outline])
            run.tests["a1_clips"] = float(len(outline))
        if low:
            run.tests["a5_before"] = mean([t["before"] for t in low])
            run.tests["a5_after"] = mean([t["after"] for t in low])
            run.tests["a5_clips"] = float(len(low))
        if a4:
            run.tests["a4_before"] = mean([b for b, _ in a4])
            run.tests["a4_after"] = mean([a for _, a in a4])
            run.tests["a4_clips"] = float(len(a4))
        for name in ("a1", "a4", "a5"):
            if f"{name}_before" in run.tests:
                before, after = run.tests[f"{name}_before"], run.tests[f"{name}_after"]
                emit({"type": "test", "model": runner.spec.id, "test": name, "before": before, "after": after})

    @staticmethod
    def _score(runner: Runner, run: _ModelRun, faces: dict[int, list[Face]], width: int, height: int) -> ModelScore:
        run.clips.sort(key=lambda c: c.index)
        for clip in run.clips:
            clip.real_rank = round(rank(clip.fake, runner.ranges.real), 1)
        fake = to_float(np.mean([c.fake for c in run.clips]))
        measures: dict[str, float] = dict(run.tests)
        if run.patch_contrib:
            share = ev.residual_share(run.frame_contrib, run.patch_contrib)
            if share is not None:
                measures["residual_share"] = share
            patch_maps = {
                frame: clip_maps[t]
                for clip, clip_maps in zip(run.clips, run.patch_contrib, strict=True)
                for t, frame in enumerate(range(clip.first_frame, clip.last_frame + 1, runner.protocol.frame_stride))
            }
            if faces:
                outline, used = ev.outline_share(patch_maps, faces, width, height)
                if outline is not None:
                    measures["outline_share"] = outline
                    measures["outline_frames"] = float(used)
            measures["gate_residual"] = round(float(np.mean([g[0] for g in run.gate])), 4)
        return ModelScore(
            model=runner.spec.id,
            name=runner.spec.name,
            fake=fake,
            real=round(1 - fake, 3),
            clips=run.clips,
            context=runner.ranges.context(fake),
            frame_values_kind="contribution" if run.patch_contrib else "probability",
            device=runner.device,
            precision=runner.precision,
            device_note=runner.device_note,
            measures=measures,
            seconds=round(run.seconds, 2),
        )

    # --- probes

    def _scorer(self, model_id: str, guard: AbstractContextManager[Any] | None = None) -> ev.Scorer:
        runner = self.load(model_id)
        hold = guard if guard is not None else nullcontext()

        def score(frames: np.ndarray, n_valid: int) -> float:
            with hold:  # the GPU is held per pass, so other requests can run between passes
                out = runner.run(frames[None], np.array([n_valid], dtype=np.int64))
            return float(out["fake_prob"][0])

        return score

    def _clip(self, path: str | Path, model_id: str, clip_index: int) -> Clip:
        p = self.load(model_id).protocol
        return read_clip(path, clip_index, p.image_size, p.n_frames, p.frame_stride)

    def probe(  # noqa: PLR0913, PLR0917
        self,
        path: str | Path,
        result: VideoResult,
        model_id: str,
        clip_index: int,
        kind: str,
        occlusion: list[list[float]] | None = None,
        *,
        guard: AbstractContextManager[Any] | None = None,
    ) -> dict[str, Any]:
        """Run a faithfulness probe on one clip.

        Parameters
        ----------
        path : str or Path
            The analyzed video.
        result : VideoResult
            Its result.
        model_id : str
            Model to probe.
        clip_index : int
            Clip number.
        kind : {"frames", "patches", "areas"}
            Remove the top frames, blank the top patches (MARS only), or blank the top
            cells of the occlusion map (any model).
        occlusion : list of list of float, optional
            The clip's occlusion map for ``areas`` (computed when not given).
        guard : context manager, optional
            Held around each pass only (the server's GPU lock).

        Returns
        -------
        dict
            Probe name, count, and the score before, after, and after random removal.
        """
        clip = self._clip(path, model_id, clip_index)
        frames = clip.inputs[self.load(model_id).protocol.image_size]
        score = self._scorer(model_id, guard)
        if kind == "frames":
            values = np.asarray(result.models[model_id].clips[clip_index].frame_values)
            return ev.probe_frames(score, frames, clip.n_valid, values, seed=clip_index).as_dict()
        if kind == "patches":
            maps = result.evidence(model_id).get("patch_contrib")
            if not maps:
                raise ValueError("patch probes need MARS evidence")
            return ev.probe_patches(score, frames, clip.n_valid, maps[clip_index], seed=clip_index).as_dict()
        if kind == "areas":
            occ = np.asarray(occlusion if occlusion is not None else self.occlusion(path, model_id, clip_index))
            return ev.probe_occlusion(score, frames, clip.n_valid, occ, seed=clip_index).as_dict()
        raise ValueError("kind must be frames, patches or areas")

    def rescore(self, path: str | Path, result: VideoResult, clip_index: int, drop: list[int]) -> dict[str, float]:
        """Score one clip again with some of its shown frames removed, for every model.

        Parameters
        ----------
        path : str or Path
            The analyzed video.
        result : VideoResult
            Its result.
        clip_index : int
            Clip number.
        drop : list of int
            Runs to remove, 16 per clip (:func:`marscope.analysis.evidence.segments`).

        Returns
        -------
        dict
            Model id to the clip's ``fake_prob`` without those frames.
        """
        out = {}
        for model_id in result.models:
            clip = self._clip(path, model_id, clip_index)
            frames = clip.inputs[self.load(model_id).protocol.image_size]
            gone = ev.drop_segments(ev.segments(clip.n_valid, SHOWN_FRAMES), drop)
            out[model_id] = round(self._scorer(model_id)(*ev.remove_frames(frames, clip.n_valid, gone)), 4)
        return out

    def sweep(
        self,
        path: str | Path,
        result: VideoResult,
        model_id: str,
        clip_index: int,
        *,
        guard: AbstractContextManager[Any] | None = None,
    ) -> dict[str, Any]:
        """Score one clip as more of the model's top frames are removed, against random.

        Parameters
        ----------
        path : str or Path
            The analyzed video.
        result : VideoResult
            Its result.
        model_id : str
            Model whose own frame values rank the frames.
        clip_index : int
            Clip number.
        guard : context manager, optional
            Held around each pass only (the server's GPU lock).

        Returns
        -------
        dict
            ``order`` (runs ranked), ``top`` and ``random`` for 0 to 8 runs removed.
        """
        clip = self._clip(path, model_id, clip_index)
        frames = clip.inputs[self.load(model_id).protocol.image_size]
        values = np.asarray(result.models[model_id].clips[clip_index].frame_values)
        out = ev.sweep_frames(
            self._scorer(model_id, guard), frames, clip.n_valid, values, SHOWN_FRAMES, seed=clip_index
        )
        return {"model": model_id, "clip": clip_index, **out}

    def edit_scores(self, path: str | Path, result: VideoResult, clip_index: int, edits: Edits) -> dict[str, float]:
        """Score one clip again for every model after what-if edits.

        Parameters
        ----------
        path : str or Path
            The analyzed video.
        result : VideoResult
            Its result.
        clip_index : int
            Clip number.
        edits : Edits
            Resolution, blur, noise and JPEG quality applied to each frame.

        Returns
        -------
        dict
            Model id to the edited clip's ``fake_prob``.
        """
        if edits.identity:
            return {m: s.clips[clip_index].fake for m, s in result.models.items()}
        out = {}
        for model_id in result.models:
            p = self.load(model_id).protocol
            clip = read_clip(path, clip_index, p.image_size, p.n_frames, p.frame_stride, transform=_editor(edits))
            out[model_id] = round(self._scorer(model_id)(clip.inputs[p.image_size], clip.n_valid), 4)
        return out

    def quality_sweep(
        self,
        path: str | Path,
        result: VideoResult,
        clip_index: int,
        edits: Edits,
        qualities: tuple[int, ...] = QUALITIES,
    ) -> dict[str, list[float]]:
        """Score one clip at several JPEG qualities, the other edits fixed.

        Parameters
        ----------
        path : str or Path
            The analyzed video.
        result : VideoResult
            Its result.
        clip_index : int
            Clip number.
        edits : Edits
            The other edits (their ``jpeg`` is replaced).
        qualities : tuple of int
            JPEG qualities to score.

        Returns
        -------
        dict
            Model id to one score per quality.
        """
        out: dict[str, list[float]] = {m: [] for m in result.models}
        for q in qualities:
            step = Edits(jpeg=q, scale=edits.scale, blur=edits.blur, noise=edits.noise)
            for m, v in self.edit_scores(path, result, clip_index, step).items():
                out[m].append(v)
        return out

    def occlusion(  # noqa: PLR0913
        self,
        path: str | Path,
        model_id: str,
        clip_index: int,
        grid: int = 8,
        *,
        guard: AbstractContextManager[Any] | None = None,
    ) -> list[list[float]]:
        """Occlusion map for one clip (any model).

        Parameters
        ----------
        path : str or Path
            The analyzed video.
        model_id : str
            Model.
        clip_index : int
            Clip number.
        grid : int
            Cells per side.
        guard : context manager, optional
            Held around each pass only (the server's GPU lock), so other requests can
            run between passes. Batching passes was measured and gives no speedup.

        Returns
        -------
        list of list of float
            (grid, grid) drop in ``fake_prob`` per blanked cell.
        """
        clip = self._clip(path, model_id, clip_index)
        frames = clip.inputs[self.load(model_id).protocol.image_size]
        result = ev.occlusion_map(self._scorer(model_id, guard), frames, clip.n_valid, grid)
        return [[round(float(v), 4) for v in row] for row in result]
