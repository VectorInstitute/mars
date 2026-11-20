"""Checks alongside the models: input scope, bitrate, faces and content credentials."""

import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from marscope.schema import Credentials, FaceSummary


log = logging.getLogger(__name__)

GRID = 24  # patches per side in the models' attribution grid
MIN_FACE_PATCHES = 5.0  # below this, "on the face outline" and "on the face" are the same
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/"
    "blaze_face_short_range.tflite"
)
FACE_MODEL_SHA256 = "b4578f35940bf5a1a655214a1cce5cab13eba73c1297cd78e1a04c2380b0152f"
AI_SOURCE_TYPES = ("trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia", "algorithmicmedia")
C2PA_FORMATS = {".mp4": "video/mp4", ".m4v": "video/x-m4v", ".mov": "video/quicktime", ".avi": "video/avi"}


@dataclass(frozen=True)
class Face:
    """One detected face, in pixels of the original frame."""

    x: float
    y: float
    width: float
    height: float
    score: float
    keypoints: tuple[tuple[float, float], ...]


def cache_dir() -> Path:
    """Return the cache folder (``$MARSCOPE_CACHE``, else ``$XDG_CACHE_HOME/marscope``).

    Returns
    -------
    Path
        The folder, created if missing.
    """
    root = os.environ.get("MARSCOPE_CACHE") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")),
        "marscope",
    )
    path = Path(root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def face_model_path() -> Path:
    """Download the BlazeFace model once and check its hash.

    Returns
    -------
    Path
        Local ``.tflite`` file.

    Raises
    ------
    RuntimeError
        If the download does not match the expected hash.
    """
    path = cache_dir() / "blaze_face_short_range.tflite"
    if not path.exists() or _sha256(path) != FACE_MODEL_SHA256:
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(FACE_MODEL_URL, tmp)  # noqa: S310 - fixed https URL
        if _sha256(tmp) != FACE_MODEL_SHA256:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("the face model download does not match its expected hash")
        tmp.replace(path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FaceFinder:
    """MediaPipe FaceDetector (BlazeFace short range), run on full-resolution frames.

    Parameters
    ----------
    min_confidence : float
        Detection confidence cut-off (a policy value).
    """

    def __init__(self, min_confidence: float = 0.5) -> None:
        os.environ.setdefault("GLOG_minloglevel", "2")  # MediaPipe logs every model load otherwise
        import mediapipe as mp  # noqa: PLC0415 - heavy import, only when faces are checked
        from mediapipe.tasks.python import vision  # noqa: PLC0415
        from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: PLC0415

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(face_model_path())),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=min_confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def detect(self, rgb: np.ndarray) -> list[Face]:
        """Find faces in one frame.

        Parameters
        ----------
        rgb : numpy.ndarray
            (H, W, 3) uint8 RGB.

        Returns
        -------
        list of Face
            Faces, largest first.
        """
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        faces = []
        for det in self._detector.detect(image).detections:
            box = det.bounding_box
            score = det.categories[0].score if det.categories else 0.0
            keypoints = tuple((float(k.x) * w, float(k.y) * h) for k in det.keypoints)
            faces.append(Face(box.origin_x, box.origin_y, box.width, box.height, float(score), keypoints))
        return sorted(faces, key=lambda f: f.width * f.height, reverse=True)

    def close(self) -> None:
        """Release the detector."""
        self._detector.close()


def face_width_patches(face: Face, frame_width: int) -> float:
    """Width of a face in patches of the 24 x 24 grid after the square resize.

    Parameters
    ----------
    face : Face
        The face.
    frame_width : int
        Original frame width in pixels.

    Returns
    -------
    float
        Width in patches.

    Examples
    --------
    >>> face_width_patches(Face(0, 0, 400, 400, 1.0, ()), 1920)
    5.0
    """
    return face.width / frame_width * GRID


def summarize_faces(faces_per_frame: dict[int, list[Face]], frame_width: int) -> FaceSummary:
    """Summarize detections over the kept frames.

    Parameters
    ----------
    faces_per_frame : dict of int to list of Face
        Detections per kept frame index.
    frame_width : int
        Original frame width.

    Returns
    -------
    FaceSummary
        Counts and the median face width.
    """
    with_face = [faces for faces in faces_per_frame.values() if faces]
    widths = [face_width_patches(faces[0], frame_width) for faces in with_face]
    return FaceSummary(
        frames_checked=len(faces_per_frame),
        frames_with_face=len(with_face),
        frames_with_several=sum(1 for faces in with_face if len(faces) > 1),
        usable_frames=sum(1 for w in widths if w >= MIN_FACE_PATCHES),
        median_width_patches=round(float(np.median(widths)), 2) if widths else None,
    )


def bits_per_pixel(size_bytes: int, n_frames: int, width: int, height: int) -> float | None:
    """Average bits per pixel, from the file size and the decoded frame count.

    The bitrate metadata is often missing or wrong, so it is not used.

    Parameters
    ----------
    size_bytes : int
        File size.
    n_frames : int
        Decoded frames.
    width, height : int
        Frame size.

    Returns
    -------
    float or None
        Bits per pixel, or None if there are no frames.

    Examples
    --------
    >>> bits_per_pixel(1_000_000, 100, 1000, 100)
    0.8
    """
    pixels = n_frames * width * height
    if pixels <= 0:
        return None
    return round(size_bytes * 8 / pixels, 4)


def scope_warnings(n_kept_frames: int, n_frames_per_clip: int, width: int, height: int) -> list[str]:
    """Ways a video falls outside what the models handle well.

    Parameters
    ----------
    n_kept_frames : int
        Frames kept after the stride.
    n_frames_per_clip : int
        Clip length (64).
    width, height : int
        Frame size.

    Returns
    -------
    list of str
        Reason keys: ``short`` (under one clip), ``low_resolution`` (short side under
        336 px, so frames are upscaled), ``wide`` (aspect ratio over 2:1, distorted by
        the square resize).

    Examples
    --------
    >>> scope_warnings(30, 64, 320, 180)
    ['short', 'low_resolution']
    """
    reasons = []
    if n_kept_frames < n_frames_per_clip:
        reasons.append("short")
    if min(width, height) < 336:
        reasons.append("low_resolution")
    if max(width, height) > 2 * min(width, height):
        reasons.append("wide")
    return reasons


def read_credentials(path: str | Path) -> Credentials:
    """Read C2PA content credentials.

    Parameters
    ----------
    path : str or Path
        Video file.

    Returns
    -------
    Credentials
        ``found`` with a summary, ``absent``, ``unsupported`` (for example WebM) or
        ``error``.
    """
    path = Path(path)
    if path.suffix.lower() not in C2PA_FORMATS:
        return Credentials(status="unsupported", detail=f"{path.suffix or 'this format'} cannot carry C2PA credentials")
    try:
        import c2pa  # noqa: PLC0415

        reader = c2pa.Reader(str(path))
        try:
            store = json.loads(reader.json())
        finally:
            reader.close()
    except Exception as error:  # noqa: BLE001 - the library raises one error type for every case
        message = str(error)
        if "manifestnotfound" in message.replace(" ", "").lower() or "no manifest" in message.lower():
            return Credentials(status="absent")
        log.info("C2PA read failed: %s", message)
        return Credentials(status="error", detail=message[:200])
    return summarize_credentials(store)


def summarize_credentials(store: dict[str, Any]) -> Credentials:
    """Summarize a C2PA manifest store.

    Parameters
    ----------
    store : dict
        Parsed manifest store (``Reader.json()``).

    Returns
    -------
    Credentials
        Status ``found``, whether it declares AI-generated content, and a one-line
        summary.

    Examples
    --------
    >>> s = {
    ...     "active_manifest": "m",
    ...     "manifests": {
    ...         "m": {
    ...             "claim_generator": "Gen/1.0",
    ...             "assertions": [
    ...                 {
    ...                     "data": {
    ...                         "actions": [
    ...                             {
    ...                                 "digitalSourceType": "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
    ...                             }
    ...                         ]
    ...                     }
    ...                 }
    ...             ],
    ...         }
    ...     },
    ... }
    >>> summarize_credentials(s).declares_ai
    True
    """
    active = store.get("active_manifest")
    manifest = store.get("manifests", {}).get(active, {}) if active else {}
    text = json.dumps(manifest).lower()
    declares_ai = any(t in text for t in AI_SOURCE_TYPES)
    generator = manifest.get("claim_generator") or next(
        (g.get("name") for g in manifest.get("claim_generator_info", []) if isinstance(g, dict)),
        None,
    )
    issuer = (manifest.get("signature_info") or {}).get("issuer")
    parts = [
        p for p in (f"made with {generator}" if generator else None, f"signed by {issuer}" if issuer else None) if p
    ]
    summary = "; ".join(parts) or "content credentials present"
    if declares_ai:
        summary += "; declares AI-generated content"
    return Credentials(status="found", declares_ai=declares_ai, summary=summary)
