"""The local gateway: REST API, progress stream and the bundled web UI in one app.

Locally it binds to 127.0.0.1 and runs one job at a time; the video stays on the machine
and is deleted when the job closes or expires.
"""

import io
import json
import logging
import queue
import re
import shutil
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from marscope.analysis.ranges import UsualRanges
from marscope.explain.ask import CHIPS, ask, chip_answer
from marscope.explain.policy import JURISDICTIONS, apply_policy, assess, load_policy, references
from marscope.explain.report import Report, build_report, llm_settings, shown_frames, to_html, to_markdown, to_pdf, tr
from marscope.models.registry import MODEL_ORDER, MODELS, parse_models
from marscope.models.runner import download
from marscope.models.scheduler import free_vram_gb
from marscope.pipeline import QUALITIES, MARScope, package_version
from marscope.schema import Experiment, VideoResult
from marscope.video.edits import Edits, apply_edits
from marscope.video.preprocess import MAX_LONG_SIDE, MAX_SECONDS, MAX_SHORT_SIDE, describe_read_error, read_meta


log = logging.getLogger(__name__)

MAX_EXPERIMENTS = 20
FLAT_MAP = 0.005  # an occlusion map whose largest change is below this shows nothing (as the UI)
TOP_FRAMES = 4  # frames the "hide its top frames" test hides (as the UI)

WEB = files("marscope").joinpath("web")
# Video AUROC (%) of the three released checkpoints, from their evaluation runs
# (set averages 91.43 MARS, 88.46 GenD 10 epochs, 90.03 DFD-HR, as on the model cards).
BENCHMARK: dict[str, Any] = json.loads(files("marscope.models").joinpath("benchmark.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Limits:
    """Upload limits, checked before a job is created. All configurable."""

    max_bytes: int = 500 * 1024 * 1024
    max_seconds: float = MAX_SECONDS
    max_long_side: int = MAX_LONG_SIDE
    max_short_side: int = MAX_SHORT_SIDE
    job_ttl_seconds: float = 3600.0


@dataclass
class Job:
    """One analysis job and its temporary files."""

    id: str
    folder: Path
    video: Path
    models: list[str]
    jurisdiction: str | None
    language: str
    status: str = "queued"
    error: str | None = None
    result: VideoResult | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    touched: float = field(default_factory=time.time)
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.Condition = field(default_factory=threading.Condition)
    experiments: dict[str, Experiment] = field(default_factory=dict)
    reports: dict[tuple[str, bool, int], Report] = field(default_factory=dict)
    cache: dict[str, dict[str, Any]] = field(default_factory=dict)  # sweeps, probes and maps, computed once
    inflight: dict[str, threading.Event] = field(default_factory=dict)  # tests and maps being computed
    version: int = 0

    def record(self, key: str, experiment: Experiment) -> None:
        """Keep a test run of this session for the report (latest per key, at most 20).

        Parameters
        ----------
        key : str
            What makes two runs the same test (a repeat replaces the earlier run).
        experiment : Experiment
            The test and the scores measured.
        """
        with self.lock:
            self.experiments.pop(key, None)
            self.experiments[key] = experiment
            while len(self.experiments) > MAX_EXPERIMENTS:
                self.experiments.pop(next(iter(self.experiments)))
            self.changed()

    def changed(self) -> None:
        """Forget built reports after the experiments or the notes change."""
        self.version += 1
        self.reports.clear()

    def emit(self, event: dict[str, Any]) -> None:
        """Record a progress event and wake the streams.

        Parameters
        ----------
        event : dict
            The event.
        """
        with self.lock:
            if event.get("type") == "stage":
                self.status = str(event.get("stage", self.status))
            self.events.append(event)
            self.lock.notify_all()


class ProbeRequest(BaseModel):
    """Body of a probe request."""

    model: str
    clip: int = Field(ge=0)
    kind: Literal["frames", "patches", "areas"]
    record: bool = Field(default=True, description="List it in the report (false for a background pre-run)")


class RescoreRequest(BaseModel):
    """Body of a re-score request: runs of the clip to remove (16 per clip)."""

    clip: int = Field(ge=0)
    drop: list[Annotated[int, Field(ge=0, lt=16)]] = Field(default_factory=list, max_length=16)


class EditRequest(BaseModel):
    """Body of a what-if request: edits applied to every frame of one clip."""

    clip: int = Field(ge=0)
    jpeg: int = Field(100, ge=10, le=100)
    scale: int = Field(100, ge=25, le=100)
    blur: float = Field(0.0, ge=0, le=3)
    noise: float = Field(0.0, ge=0, le=0.1)

    def edits(self) -> Edits:
        """Return the edits.

        Returns
        -------
        Edits
            The requested edits.
        """
        return Edits(jpeg=self.jpeg, scale=self.scale, blur=self.blur, noise=self.noise)


class SweepRequest(BaseModel):
    """Body of a sweep request."""

    model: str
    clip: int = Field(ge=0)
    record: bool = Field(default=True, description="List it in the report (false for a background pre-run)")


class OcclusionRequest(BaseModel):
    """Body of an occlusion-map request."""

    model: str
    clip: int = Field(ge=0)
    record: bool = Field(default=True, description="List it in the report (false for a background pre-run)")


class ReportRequest(BaseModel):
    """Body of a report request."""

    format: Literal["json", "markdown", "html", "pdf"] = "json"
    use_llm: bool = True
    language: Literal["en", "fr"] = "en"
    frame: int | None = Field(default=None, ge=0, description="Kept frame to show in HTML and PDF (optional)")


class AskRequest(BaseModel):
    """Body of a question: a suggested question (``chip``) or a free one."""

    chip: Literal["disagree", "tests", "certain", "verdict", "deepfake"] | None = None
    question: str | None = Field(default=None, min_length=1, max_length=500)
    language: Literal["en", "fr"] = "en"


class NotesRequest(BaseModel):
    """Body of a request to re-apply the notes for another region or language."""

    jurisdiction: Literal["eu", "ca", "us", "all"] | None = None
    language: Literal["en", "fr"] = "en"


GPU_OUT_OF_MEMORY = (
    "the GPU ran out of memory: close other programs using it, pick fewer models, or start MARScope with --sequential"
)


def once(job: Job, key: str, compute: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Compute a test or map once per job: cached after, and shared while it runs.

    The computation holds the GPU per pass (``guard`` in the pipeline), so the user's
    other requests run between passes. A second request for the same key while it
    runs waits for it instead of computing it again.

    Parameters
    ----------
    job : Job
        The job, whose cache holds the result.
    key : str
        What makes two requests the same.
    compute : callable
        Computes the result.

    Returns
    -------
    dict
        The result.
    """
    while True:
        with job.lock:
            cached = job.cache.get(key)
            if cached is not None:
                return cached
            running = job.inflight.get(key)
            mine = running is None
            if running is None:
                running = job.inflight[key] = threading.Event()
        if not mine:
            running.wait()
            continue  # cached now, or the other request failed and this one computes it
        try:
            out = compute()
            job.cache[key] = out
            return out
        finally:
            with job.lock:
                job.inflight.pop(key, None)
            running.set()


def friendly_error(error: BaseException) -> str:
    """Turn an error from the models into a message a person can act on.

    Parameters
    ----------
    error : BaseException
        What the analysis or a test raised.

    Returns
    -------
    str
        The message (the error's own text when there is nothing clearer to say).

    Examples
    --------
    >>> message = friendly_error(RuntimeError("CUDA failure 2: out of memory"))
    >>> message.startswith("the GPU ran out of memory")
    True
    """
    text = str(error) or type(error).__name__
    lowered = text.lower()
    if "out of memory" in lowered or "failed to allocate" in lowered or "cudaerrormemoryallocation" in lowered:
        return GPU_OUT_OF_MEMORY
    return text


def safe_name(filename: str | None) -> str:
    """Keep an upload's own name, reduced to safe characters.

    Parameters
    ----------
    filename : str or None
        Name sent by the browser.

    Returns
    -------
    str
        A plain file name with an extension.

    Examples
    --------
    >>> safe_name("../My clip (1).MP4")
    'My_clip_1.mp4'
    >>> safe_name(None)
    'video.mp4'
    """
    name = Path(filename or "video.mp4").name
    stem, suffix = Path(name).stem, Path(name).suffix.lower() or ".mp4"
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "video"
    return f"{stem[:80]}{suffix if re.fullmatch(r'\.[a-z0-9]{1,5}', suffix) else '.mp4'}"


def sniff_video(head: bytes) -> str | None:
    """Recognize a video container from its first bytes, not its extension.

    Parameters
    ----------
    head : bytes
        At least the first 12 bytes of the file.

    Returns
    -------
    str or None
        ``mp4`` (MP4 and MOV share the ISO base media format), ``webm``, or None.

    Examples
    --------
    >>> sniff_video(b"\\x00\\x00\\x00\\x18ftypmp42")
    'mp4'
    >>> sniff_video(b"\\x1a\\x45\\xdf\\xa3" + b"\\x00" * 8)
    'webm'
    >>> sniff_video(b"GIF89a......") is None
    True
    """
    if len(head) >= 12 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free"):
        return "mp4"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    return None


class JobManager:
    """Queue jobs and run them one at a time on a worker thread.

    Parameters
    ----------
    toolkit : MARScope
        The loaded toolkit (shared by all jobs).
    limits : Limits
        Upload limits and job expiry.
    """

    def __init__(self, toolkit: MARScope, limits: Limits) -> None:
        self.toolkit = toolkit
        self.limits = limits
        self.jobs: dict[str, Job] = {}
        self.gpu = threading.Lock()  # one analysis, probe or load at a time
        self._queue: queue.Queue[str] = queue.Queue()
        self._root = Path(tempfile.mkdtemp(prefix="marscope-jobs-"))
        threading.Thread(target=self._work, daemon=True, name="marscope-worker").start()
        threading.Thread(target=self._expire, daemon=True, name="marscope-expiry").start()

    def submit(self, upload: UploadFile, models: list[str], jurisdiction: str | None, language: str) -> Job:
        """Validate an upload and queue it.

        Parameters
        ----------
        upload : UploadFile
            The uploaded video.
        models : list of str
            Models to run.
        jurisdiction : str or None
            Region for context notes.
        language : str
            Note language.

        Returns
        -------
        Job
            The queued job.

        Raises
        ------
        HTTPException
            400 or 413 when the upload is rejected; a rejected upload is never kept.
        """
        job_id = uuid.uuid4().hex[:12]
        folder = self._root / job_id
        folder.mkdir()
        video = folder / safe_name(upload.filename)
        try:
            size = 0
            with video.open("wb") as out:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.limits.max_bytes:
                        raise HTTPException(413, f"the file is larger than {self.limits.max_bytes // 2**20} MB")
                    out.write(chunk)
            with video.open("rb") as f:
                kind = sniff_video(f.read(16))
            if kind is None:
                raise HTTPException(400, "the file is not an MP4, MOV or WebM video")
            try:
                meta = read_meta(video)
            except Exception as error:
                raise HTTPException(400, describe_read_error(error)) from error
            if meta.duration_s > self.limits.max_seconds:
                raise HTTPException(400, f"the video is longer than {self.limits.max_seconds:.0f} seconds")
            long_side, short_side = max(meta.width, meta.height), min(meta.width, meta.height)
            if long_side > self.limits.max_long_side or short_side > self.limits.max_short_side:
                raise HTTPException(400, "the video is larger than 1080p")
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        job = Job(job_id, folder, video, models, jurisdiction, language)
        job.events.append({"type": "stage", "stage": "queued", "name": upload.filename})
        self.jobs[job_id] = job
        self._queue.put(job_id)
        return job

    def get(self, job_id: str) -> Job:
        """Return a job or raise 404.

        Parameters
        ----------
        job_id : str
            Job id.

        Returns
        -------
        Job
            The job.
        """
        job = self.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "no such job (it may have expired)")
        job.touched = time.time()
        return job

    def result(self, job_id: str) -> tuple[Job, VideoResult]:
        """Return a finished job and its result, or raise 409 while it runs.

        Parameters
        ----------
        job_id : str
            Job id.

        Returns
        -------
        tuple of (Job, VideoResult)
            The job and its result.
        """
        job = self.get(job_id)
        if job.result is None:
            raise HTTPException(409, job.error or f"the job is {job.status}")
        return job, job.result

    def delete(self, job_id: str) -> None:
        """Delete a job's video and result now.

        Parameters
        ----------
        job_id : str
            Job id.
        """
        job = self.jobs.pop(job_id, None)
        if job is not None:
            shutil.rmtree(job.folder, ignore_errors=True)

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            try:
                with self.gpu:
                    job.result = self.toolkit.analyze(
                        job.video,
                        jurisdiction=job.jurisdiction,
                        language=job.language,
                        models=job.models,
                        progress=job.emit,
                        frames_dir=job.folder / "frames",
                    )
                job.emit({"type": "stage", "stage": "done"})
            except Exception as error:  # noqa: BLE001 - reported to the client as a failed job
                log.exception("job %s failed", job_id)
                job.error = friendly_error(error)
                job.emit({"type": "stage", "stage": "failed", "error": job.error})
            finally:
                with job.lock:
                    job.done.set()
                    job.lock.notify_all()

    def _expire(self) -> None:
        while True:
            time.sleep(60)
            now = time.time()
            for job_id, job in list(self.jobs.items()):
                if job.done.is_set() and now - job.touched > self.limits.job_ttl_seconds:
                    log.info("job %s expired", job_id)
                    self.delete(job_id)


def create_app(toolkit: MARScope | None = None, limits: Limits | None = None) -> FastAPI:  # noqa: PLR0915
    """Build the FastAPI app.

    Parameters
    ----------
    toolkit : MARScope, optional
        A toolkit to share; by default one with all three models (loaded on demand).
    limits : Limits, optional
        Upload limits.

    Returns
    -------
    FastAPI
        The app: web UI at ``/`` and ``/app``, API under the same port.
    """
    toolkit = toolkit or MARScope()
    manager = JobManager(toolkit, limits or Limits())
    app = FastAPI(title="MARScope", version=package_version(), docs_url="/api/docs", redoc_url=None)
    app.state.manager = manager
    app.state.ready = True  # serve() sets False while it preloads models in the background
    app.state.loading_model = None

    @app.get("/app", response_class=HTMLResponse, include_in_schema=False)
    def application() -> HTMLResponse:
        return HTMLResponse(WEB.joinpath("app.html").read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        """Status, device and free GPU memory."""
        free = free_vram_gb(toolkit.device_id)
        settings = llm_settings()
        return {
            "status": "ok" if app.state.ready else "loading",
            "loading_model": app.state.loading_model,
            "version": package_version(),
            "gpu": free is not None,
            "free_vram_gb": round(free, 2) if free is not None else None,
            "loaded": toolkit.loaded(),
            "llm": settings is not None,
            "llm_model": settings["model"] if settings else None,
        }

    @app.get("/models")
    def models() -> list[dict[str, Any]]:
        """List the models, which are loaded, and their measured cost (fp16)."""
        loaded = set(toolkit.loaded())
        return [
            {
                "id": m,
                "name": MODELS[m].name,
                "repo": MODELS[m].repo,
                "card": f"https://huggingface.co/{MODELS[m].repo}",
                "loaded": m in loaded,
                "built_in_attribution": MODELS[m].built_in_attribution,
                **serving_summary(m),
                "typical": typical_ranges(m),
                "benchmark": BENCHMARK["models"].get(m),
            }
            for m in MODEL_ORDER
        ]

    @app.get("/benchmark")
    def benchmark() -> dict[str, Any]:
        """Return video AUROC (%) by test set and generator for the released models."""
        return BENCHMARK

    @app.get("/references")
    def all_references() -> list[dict[str, str]]:
        """Every reference the policy can cite."""
        return references(list(load_policy()["references"]))

    @app.post("/models/{model_id}/load")
    def load_model(model_id: str) -> dict[str, Any]:
        """Load a model ahead of time (the UI calls this when a box is ticked)."""
        try:
            parse_models(model_id)
        except ValueError as error:
            raise HTTPException(404, str(error)) from error
        with manager.gpu:
            runner = toolkit.load(model_id)
        return {
            "id": model_id,
            "device": runner.device,
            "precision": runner.precision,
            "device_note": runner.device_note,
            "peak_vram_gb": runner.peak_vram_gb(),
        }

    @app.post("/jobs", status_code=202)
    def create_job(
        video: UploadFile = File(...),  # noqa: B008 - FastAPI's way of declaring uploads
        models: str = Form("mars"),
        jurisdiction: str = Form(""),
        language: str = Form("en"),
    ) -> dict[str, str]:
        """Upload a video and start a job. Returns at once with the job id."""
        try:
            model_ids = parse_models(models)
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        region = jurisdiction.strip().lower() or None
        if region is not None and region not in JURISDICTIONS:
            raise HTTPException(400, f"jurisdiction must be one of {', '.join(JURISDICTIONS)}")
        if language not in ("en", "fr"):
            raise HTTPException(400, "language must be en or fr")
        job = manager.submit(video, model_ids, region, language)
        return {"id": job.id, "status": job.status}

    @app.get("/jobs/{job_id}/events")
    def events(job_id: str) -> StreamingResponse:
        """Progress as server-sent events, until the job is done or failed."""
        job = manager.get(job_id)

        def stream() -> Iterator[str]:
            sent = 0
            while True:
                with job.lock:
                    if sent >= len(job.events) and not job.done.is_set():
                        job.lock.wait(timeout=15)
                    pending = job.events[sent:]
                    finished = job.done.is_set()
                for event in pending:
                    yield f"data: {json.dumps(event)}\n\n"
                sent += len(pending)
                if finished:
                    return
                if not pending:
                    yield ": keep-alive\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        """Status, and the result when done: scores, evidence measures and notes."""
        job = manager.get(job_id)
        body: dict[str, Any] = {"id": job.id, "status": job.status, "error": job.error}
        if job.result is not None:
            body["result"] = job.result.model_dump(mode="json")
            body["references"] = references(sorted({k for n in job.result.notes for k in n.references}))
            body["ranges"] = {
                m: {k: [round(float(v), 5) for v in ev[f"{k}_percentiles"]] for k in ("real", "fake")}
                for m in job.result.models
                if "real_percentiles" in (ev := job.result.evidence(m))
            }
        return body

    @app.get("/jobs/{job_id}/frames/{frame}.jpg")
    def frame_image(job_id: str, frame: int) -> FileResponse:
        """Return a kept frame as a JPEG thumbnail."""
        path = manager.get(job_id).folder / "frames" / f"{frame:06d}.jpg"
        if not path.exists():
            raise HTTPException(404, "no such frame")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/jobs/{job_id}/maps/{model_id}/{clip}")
    def patch_maps(job_id: str, model_id: str, clip: int) -> dict[str, Any]:
        """Return MARS patch maps for one clip: one per real frame, and their sum."""
        _, result = manager.result(job_id)
        maps = result.evidence(model_id).get("patch_contrib")
        if not maps or clip >= len(maps):
            raise HTTPException(404, "no patch maps for this model and clip")
        arr: np.ndarray = maps[clip]
        return {
            "frames": np.round(arr, 5).tolist(),
            "clip": np.round(arr.sum(axis=0), 5).tolist(),
            "scale": float(np.abs(arr).max() or 1.0),
        }

    @app.post("/jobs/{job_id}/probe")
    def probe(job_id: str, body: ProbeRequest) -> dict[str, Any]:
        """Remove top frames or blank top patches of a clip, against random removal."""
        job, result = manager.result(job_id)
        if body.model not in result.models or body.clip >= len(result.models[body.model].clips):
            raise HTTPException(404, "no such model or clip in this result")
        key = f"probe:{body.kind}:{body.model}:{body.clip}"
        occ = occlusion_of(job, body.model, body.clip) if body.kind == "areas" else None
        try:
            out = once(
                job,
                key,
                lambda: toolkit.probe(
                    job.video, result, body.model, body.clip, body.kind, occlusion=occ, guard=manager.gpu
                ),
            )
        except ValueError as error:
            raise HTTPException(400, str(error)) from error
        if not body.record:
            return out
        kind: Literal["areas_test", "frames_test"] = "frames_test" if body.kind == "frames" else "areas_test"
        job.record(
            f"{kind}:{body.model}:{body.clip}",
            Experiment(
                kind=kind,
                clip=body.clip,
                model=body.model,
                hidden=int(out["count"]),
                before={body.model: out["before"]},
                after={body.model: out["after"]},
                random_after=out["random_after"],
            ),
        )
        return out

    @app.post("/jobs/{job_id}/rescore")
    def rescore(job_id: str, body: RescoreRequest) -> dict[str, Any]:
        """Score one clip again for every model, with some shown frames removed."""
        job, result = manager.result(job_id)
        if body.clip >= result.video.n_clips:
            raise HTTPException(404, "no such clip in this result")
        if not body.drop:
            return {
                "clip": body.clip,
                "drop": [],
                "scores": {m: s.clips[body.clip].fake for m, s in result.models.items()},
            }
        with manager.gpu:
            scores = toolkit.rescore(job.video, result, body.clip, sorted(set(body.drop)))
        drop = sorted(set(body.drop))
        job.record(
            f"frames:{body.clip}",
            Experiment(
                kind="frames",
                clip=body.clip,
                hidden=len(drop),
                frames=shown_frames(result, body.clip, drop),
                before={m: s.clips[body.clip].fake for m, s in result.models.items()},
                after=scores,
            ),
        )
        return {"clip": body.clip, "drop": drop, "scores": scores}

    @app.post("/jobs/{job_id}/edit")
    def edit(job_id: str, body: EditRequest) -> dict[str, Any]:
        """Score one clip again for every model after what-if edits."""
        job, result = manager.result(job_id)
        if body.clip >= result.video.n_clips:
            raise HTTPException(404, "no such clip in this result")
        with manager.gpu:
            scores = toolkit.edit_scores(job.video, result, body.clip, body.edits())
        settings = {k: float(v) for k, v in body.edits().as_dict().items()}
        if body.jpeg < 100 or body.scale < 100 or body.blur > 0 or body.noise > 0:
            job.record(
                f"edit:{body.clip}:{sorted(settings.items())}",
                Experiment(
                    kind="edit",
                    clip=body.clip,
                    settings=settings,
                    before={m: s.clips[body.clip].fake for m, s in result.models.items()},
                    after=scores,
                ),
            )
        return {"clip": body.clip, "edits": body.edits().as_dict(), "scores": scores}

    @app.post("/jobs/{job_id}/quality-sweep")
    def quality_sweep(job_id: str, body: EditRequest) -> dict[str, Any]:
        """Score one clip at several JPEG qualities, the other edits fixed."""
        job, result = manager.result(job_id)
        if body.clip >= result.video.n_clips:
            raise HTTPException(404, "no such clip in this result")
        with manager.gpu:
            scores = toolkit.quality_sweep(job.video, result, body.clip, body.edits())
        return {"clip": body.clip, "qualities": list(QUALITIES), "scores": scores}

    @app.get("/jobs/{job_id}/edited/{frame}.jpg")
    def edited_image(  # noqa: PLR0913, PLR0917
        job_id: str,
        frame: int,
        jpeg: int = Query(100, ge=10, le=100),
        scale: int = Query(100, ge=25, le=100),
        blur: float = Query(0.0, ge=0, le=3),
        noise: float = Query(0.0, ge=0, le=0.1),
    ) -> Response:
        """Return a preview of a kept frame with what-if edits (from its thumbnail)."""
        job, result = manager.result(job_id)
        path = job.folder / "frames" / f"{frame:06d}.jpg"
        if not path.exists():
            raise HTTPException(404, "no such frame")
        with Image.open(path) as img:
            rgb = np.asarray(img.convert("RGB"))
        out = apply_edits(
            rgb, Edits(jpeg, scale, blur, noise), seed=frame, pixel_scale=rgb.shape[0] / result.video.height
        )
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="JPEG", quality=95)
        return Response(buf.getvalue(), media_type="image/jpeg")

    @app.post("/jobs/{job_id}/sweep")
    def sweep(job_id: str, body: SweepRequest) -> dict[str, Any]:
        """Score a clip as more of the model's top frames go, versus random."""
        job, result = manager.result(job_id)
        if body.model not in result.models or body.clip >= len(result.models[body.model].clips):
            raise HTTPException(404, "no such model or clip in this result")
        out = once(
            job,
            f"sweep:{body.model}:{body.clip}",
            lambda: toolkit.sweep(job.video, result, body.model, body.clip, guard=manager.gpu),
        )
        k = min(TOP_FRAMES, len(out["top"]) - 1)
        if body.record and k > 0:
            job.record(
                f"frames_test:{body.model}:{body.clip}",
                Experiment(
                    kind="frames_test",
                    clip=body.clip,
                    model=body.model,
                    hidden=k,
                    frames=shown_frames(result, body.clip, list(out["order"][:k])),
                    before={body.model: float(out["top"][0])},
                    after={body.model: float(out["top"][k])},
                    random_after=float(out["random"][k]),
                ),
            )
        return out

    @app.post("/jobs/{job_id}/occlusion")
    def occlusion(job_id: str, body: OcclusionRequest) -> dict[str, Any]:
        """Compute an 8 x 8 occlusion map for one clip (64 passes; slow on DFD-HR)."""
        job, result = manager.result(job_id)
        if body.model not in result.models or body.clip >= len(result.models[body.model].clips):
            raise HTTPException(404, "no such model or clip in this result")
        grid = occlusion_of(job, body.model, body.clip)
        if body.record:
            flat = max((v for row in grid for v in row), default=0.0) < FLAT_MAP
            job.record(
                f"occlusion:{body.model}:{body.clip}",
                Experiment(kind="occlusion", clip=body.clip, model=body.model, flat=flat),
            )
        return {"model": body.model, "clip": body.clip, "map": grid}

    def occlusion_of(job: Job, model_id: str, clip: int) -> list[list[float]]:
        """Return a clip's occlusion map, computed once per job and cached."""
        out = once(
            job,
            f"occlusion:{model_id}:{clip}",
            lambda: {"map": toolkit.occlusion(job.video, model_id, clip, guard=manager.gpu)},
        )
        grid: list[list[float]] = out["map"]
        return grid

    @app.post("/jobs/{job_id}/report", response_model=None)
    def report(job_id: str, body: ReportRequest) -> Response | dict[str, Any]:
        """Build the report, with the session's tests, as JSON, Markdown, HTML or PDF.

        The built report is kept until the tests or the notes change, so every export
        of one report has the same summary (the LLM is asked once).
        """
        job, result = manager.result(job_id)
        key = (body.language, body.use_llm, job.version)
        built = job.reports.get(key)
        if built is None:
            experiments = list(job.experiments.values())
            built = build_report(result, use_llm=body.use_llm, experiments=experiments, language=body.language)
            job.reports[key] = built
        stem = Path(result.video.name).stem or "video"
        if body.frame is not None and body.format in ("html", "pdf"):
            built = with_frame(built, job, result, body.frame)
        if body.format == "markdown":
            return Response(to_markdown(built), media_type="text/markdown", headers=_download(f"{stem}-report.md"))
        if body.format == "html":
            return HTMLResponse(to_html(built), headers=_download(f"{stem}-report.html"))
        if body.format == "pdf":
            return Response(to_pdf(built), media_type="application/pdf", headers=_download(f"{stem}-report.pdf"))
        return {
            "title": built.title,
            "summary": built.summary,
            "summary_source": built.summary_source,
            "by": built.by_line,
            "llm_status": built.llm_status,
            "rows": built.rows,
            "experiments": built.experiments,
            "records": [e.model_dump(mode="json") for e in job.experiments.values()],
            "notes": [n.model_dump(mode="json") for n in built.notes],
            "references": built.refs,
            "versions": built.versions,
            "llm": built.llm,
            "payload": built.payload,
            "disclaimer": built.disclaimer,
            "markdown": to_markdown(built),
        }

    @app.post("/jobs/{job_id}/ask")
    def ask_question(job_id: str, body: AskRequest) -> dict[str, Any]:
        """Answer a suggested question from the result, or a free one with the LLM."""
        job, result = manager.result(job_id)
        if (body.chip is None) == (body.question is None):
            raise HTTPException(422, f"send either chip ({', '.join(CHIPS)}) or question")
        if body.chip is not None:
            answer = chip_answer(result, body.chip, body.language)
        else:
            answer = ask(result, body.question or "", body.language, list(job.experiments.values()))
        return {
            "text": answer.text,
            "source": answer.source,
            "references": answer.references,
            "llm": answer.llm,
            "llm_status": answer.llm_status,
            "refused": answer.refused,
        }

    @app.post("/jobs/{job_id}/notes")
    def notes(job_id: str, body: NotesRequest) -> dict[str, Any]:
        """Re-apply the policy for another region or language. Scores do not change."""
        job, result = manager.result(job_id)
        result.notes = apply_policy(result, body.jurisdiction, body.language)
        result.reading = assess(result, body.language)
        result.jurisdiction = body.jurisdiction
        job.result = result
        job.changed()
        return {
            "notes": [n.model_dump(mode="json") for n in result.notes],
            "reading": result.reading.model_dump(mode="json"),
            "references": references(sorted({k for n in result.notes for k in n.references})),
        }

    @app.delete("/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        """Delete the video and the result now."""
        manager.get(job_id)
        manager.delete(job_id)
        return Response(status_code=204)

    @app.exception_handler(HTTPException)
    def http_error(_request: Any, error: HTTPException) -> JSONResponse:
        return JSONResponse({"error": error.detail}, status_code=error.status_code)

    @app.exception_handler(Exception)
    def unexpected_error(_request: Any, error: Exception) -> JSONResponse:
        log.exception("request failed")
        return JSONResponse({"error": friendly_error(error)}, status_code=500)

    # The web UI last, at the root: relative links work served and opened as files.
    app.mount("/", StaticFiles(directory=str(WEB), html=True), name="web")
    return app


def typical_ranges(model_id: str) -> dict[str, float] | None:
    """Return typical benchmark scores: real median, real 95th percentile, fake median.

    Parameters
    ----------
    model_id : str
        Model id.

    Returns
    -------
    dict or None
        ``real_p50``, ``real_p95``, ``fake_p50``, or None when the file cannot be read.
    """
    try:
        ranges = UsualRanges.from_file(download(MODELS[model_id], "usual_ranges.fp16.json"))
    except Exception as error:  # noqa: BLE001 - offline or not yet downloaded
        log.debug("ranges for %s unavailable: %s", model_id, error)
        return None
    return {
        "real_p50": round(float(ranges.real[50]), 4),
        "real_p95": round(float(ranges.real[95]), 4),
        "fake_p50": round(float(ranges.fake[50]), 4),
    }


def serving_summary(model_id: str) -> dict[str, float | None]:
    """Measured peak memory and time per clip for the fp16 file, from ``serving.json``.

    Parameters
    ----------
    model_id : str
        Model id.

    Returns
    -------
    dict
        ``peak_vram_gb`` and ``seconds_per_clip`` (None when the file cannot be read).
    """
    try:
        serving = json.loads(download(MODELS[model_id], "serving.json").read_text())
        fp16 = serving["files"]["model.fp16.onnx"]
        return {"peak_vram_gb": fp16["peak_vram_gb"]["batch1"], "seconds_per_clip": fp16["seconds_per_clip"]["batch1"]}
    except Exception as error:  # noqa: BLE001 - offline or not yet downloaded
        log.debug("serving.json for %s unavailable: %s", model_id, error)
        return {"peak_vram_gb": None, "seconds_per_clip": None}


def with_frame(report: Report, job: Job, result: VideoResult, frame: int) -> Report:
    """Return a copy of a report showing one kept frame (its thumbnail) and a caption.

    Parameters
    ----------
    report : Report
        The built report (kept unchanged in the cache).
    job : Job
        The job, whose folder holds the thumbnails.
    result : VideoResult
        Its result (to find the clip).
    frame : int
        Kept frame index in the original video.

    Returns
    -------
    Report
        The copy with ``frame`` and ``frame_caption`` set.
    """
    path = job.folder / "frames" / f"{frame:06d}.jpg"
    if not path.exists():
        raise HTTPException(404, "no such frame")
    clips = next(iter(result.models.values())).clips
    clip = next((c.index for c in clips if c.first_frame <= frame <= c.last_frame), 0)
    caption = tr(report.language, "frame_cap", n=frame + 1, c=clip + 1)
    return replace(report, frame=path.read_bytes(), frame_caption=caption)


def _download(name: str) -> dict[str, str]:
    return {"Content-Disposition": f'attachment; filename="{name}"'}


def serve(
    host: str = "127.0.0.1", port: int = 8000, toolkit: MARScope | None = None, preload: list[str] | None = None
) -> None:
    """Run the local server (``marscope --serve``).

    Parameters
    ----------
    host : str
        Interface to bind; 127.0.0.1 keeps it local.
    port : int
        Port.
    toolkit : MARScope, optional
        MARScope to use.
    preload : list of str, optional
        Models to load at start (MARS by default), with one warm-up.
    """
    import uvicorn  # noqa: PLC0415

    toolkit = toolkit or MARScope()
    app = create_app(toolkit)
    app.state.ready = False

    def warm_up() -> None:
        try:
            with app.state.manager.gpu:
                for model_id in preload if preload is not None else ["mars"]:
                    app.state.loading_model = model_id
                    log.info("loading %s", model_id)
                    toolkit.load(model_id)
        except Exception:  # noqa: BLE001 - the server still starts; jobs load models on demand
            log.exception("preloading failed")
        finally:
            app.state.loading_model = None
            app.state.ready = True

    threading.Thread(target=warm_up, daemon=True, name="marscope-warm-up").start()
    uvicorn.run(app, host=host, port=port, log_level="info")
