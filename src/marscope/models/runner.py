"""ONNX Runtime sessions for the released detectors.

The session settings are the ones ``serving.json`` was measured with; the memory numbers
the scheduler uses depend on them. ONNX Runtime falls back to the CPU silently when CUDA
libraries are missing, so the runner checks the providers after creating each session
and reports a fallback.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

from marscope.analysis.ranges import UsualRanges
from marscope.models.registry import ModelSpec
from marscope.schema import Device, Precision


log = logging.getLogger(__name__)

FILES: dict[str, tuple[str, str]] = {
    "fp16": ("model.fp16.onnx", "usual_ranges.fp16.json"),
    "fp32": ("model.onnx", "usual_ranges.json"),
}
CUDA_OPTIONS = {"arena_extend_strategy": "kSameAsRequested", "cudnn_conv_algo_search": "HEURISTIC"}


@dataclass(frozen=True)
class Protocol:
    """Input settings stored in the ONNX metadata."""

    n_frames: int
    frame_stride: int
    image_size: int
    outputs: tuple[str, ...]


def download(spec: ModelSpec, filename: str) -> Path:
    """Download one model file from its repository's main branch (cached).

    Parameters
    ----------
    spec : ModelSpec
        The model.
    filename : str
        File in the repository.

    Returns
    -------
    Path
        Local path.
    """
    return Path(hf_hub_download(spec.repo, filename))


def commit_of(path: Path) -> str:
    """Return the Hugging Face commit a downloaded file came from.

    The cache stores files under ``snapshots/<commit>/``; results print that commit.

    Parameters
    ----------
    path : Path
        A path returned by :func:`download`.

    Returns
    -------
    str
        The commit hash, or ``"unknown"`` for a file outside the cache.

    Examples
    --------
    >>> commit_of(Path("/c/models--x--y/snapshots/9baf809130e6/model.onnx"))
    '9baf809130e6'
    """
    parts = path.parts
    if "snapshots" in parts:
        i = parts.index("snapshots")
        if i + 1 < len(parts):
            return parts[i + 1]
    return "unknown"


def cpu_threads() -> int:
    """Return the number of CPUs this process may use.

    ONNX Runtime otherwise sees every core on the machine, which oversubscribes shared
    nodes.

    Returns
    -------
    int
        Usable CPUs.
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not on Linux
        return os.cpu_count() or 1


def cuda_available() -> bool:
    """Return whether this ONNX Runtime build has the CUDA execution provider.

    Returns
    -------
    bool
        True if CUDA can be requested (it may still fail to load; see :class:`Runner`).
    """
    return "CUDAExecutionProvider" in ort.get_available_providers()


def _preload_cuda_libraries() -> None:
    preload = getattr(ort, "preload_dlls", None)
    if preload is not None:
        try:
            preload()
        except Exception as error:  # noqa: BLE001 - missing wheels are reported by the provider check
            log.debug("preload_dlls failed: %s", error)


class Runner:
    """One loaded model.

    Parameters
    ----------
    spec : ModelSpec
        The model.
    device : {"auto", "cuda", "cpu"}
        ``auto`` uses CUDA when available. fp16 on the GPU, fp32 on the CPU.
    device_id : int
        GPU index.
    precision : {"auto", "fp16", "fp32"}
        ``auto`` is fp16 on the GPU and fp32 on the CPU. fp16 is never used on the CPU.
    """

    def __init__(self, spec: ModelSpec, device: str = "auto", device_id: int = 0, precision: str = "auto") -> None:
        self.spec = spec
        self.device_note: str | None = None
        want_cuda = device == "cuda" or (device == "auto" and cuda_available())
        if device == "cuda" and not cuda_available():
            self.device_note = "CUDA was requested but this ONNX Runtime build has no CUDA provider"
        session: Any = None
        if want_cuda:
            _preload_cuda_libraries()
            cuda = ("CUDAExecutionProvider", {"device_id": device_id, **CUDA_OPTIONS})
            session = self._open("fp32" if precision == "fp32" else "fp16", [cuda, "CPUExecutionProvider"])
            if "CUDAExecutionProvider" not in session.get_providers():
                self.device_note = (
                    "CUDA could not be loaded (CUDA 12 or cuDNN 9 libraries missing), so the model runs on the CPU "
                    "with the fp32 file"
                )
                log.warning(
                    "%s: %s. Install the CUDA libraries with `uv sync --extra cuda` (or `pip install marscope[cuda]`)",
                    spec.name,
                    self.device_note,
                )
                session = None
        if session is None:
            session = self._open("fp32", ["CPUExecutionProvider"])
        self.session: Any = session
        self.device: Device = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        if precision == "fp16" and self.device == "cpu" and self.device_note is None:
            self.device_note = "fp16 runs on the GPU only; on the CPU the model uses the fp32 file"
        self.protocol = self._read_protocol()
        self.ranges = UsualRanges.from_file(download(spec, FILES[self.precision][1]))
        self.serving: dict[str, Any] = json.loads(download(spec, "serving.json").read_text())
        self.run_options = ort.RunOptions()
        if self.device == "cuda":
            self.run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", f"gpu:{device_id}")

    def _open(self, precision: Precision, providers: list[Any]) -> Any:
        options = ort.SessionOptions()
        options.intra_op_num_threads = cpu_threads()
        options.log_severity_level = 3  # errors only; ORT warns about graph rewrites on every load
        path = download(self.spec, FILES[precision][0])
        self.commit = commit_of(path)
        self.precision: Precision = precision
        return ort.InferenceSession(str(path), sess_options=options, providers=providers)

    def _read_protocol(self) -> Protocol:
        meta = self.session.get_modelmeta().custom_metadata_map
        outputs = tuple(o.name for o in self.session.get_outputs())
        return Protocol(
            n_frames=int(meta["n_frames"]),
            frame_stride=int(meta["frame_stride"]),
            image_size=int(meta["image_size"]),
            outputs=outputs,
        )

    def run(self, frames: np.ndarray, n_valid: np.ndarray) -> dict[str, np.ndarray]:
        """Score a batch of clips.

        Parameters
        ----------
        frames : numpy.ndarray
            (B, n_frames, 3, S, S) float32 in [-1, 1].
        n_valid : numpy.ndarray
            (B,) int64, real frames per clip.

        Returns
        -------
        dict of str to numpy.ndarray
            Every output of the graph: ``fake_prob``, ``logits`` and, per model,
            ``frame_contrib``, ``patch_contrib``, ``gate`` (MARS) or ``frame_probs``
            (GenD, DFD-HR).
        """
        feed = {
            "frames": np.ascontiguousarray(frames, dtype=np.float32),
            "n_valid": np.asarray(n_valid, dtype=np.int64),
        }
        values = self.session.run(list(self.protocol.outputs), feed, self.run_options)
        return dict(zip(self.protocol.outputs, values, strict=True))

    def peak_vram_gb(self, batch: int = 1) -> float:
        """Return the measured peak GPU memory for this file and batch size.

        Parameters
        ----------
        batch : int
            1 or 2.

        Returns
        -------
        float
            GiB, from ``serving.json``.
        """
        return peak_vram_gb(self.serving, self.precision, batch)


def peak_vram_gb(serving: dict[str, Any], precision: Precision, batch: int = 1) -> float:
    """Read a peak from a parsed ``serving.json``.

    Parameters
    ----------
    serving : dict
        Parsed ``serving.json``.
    precision : {"fp16", "fp32"}
        Which file.
    batch : int
        1 or 2.

    Returns
    -------
    float
        GiB.

    Examples
    --------
    >>> peaks = {"peak_vram_gb": {"batch1": 2.25, "batch2": 3.43}}
    >>> s = {"files": {"model.fp16.onnx": peaks}}
    >>> peak_vram_gb(s, "fp16")
    2.25
    """
    return float(serving["files"][FILES[precision][0]]["peak_vram_gb"][f"batch{batch}"])
