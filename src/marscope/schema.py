"""Result types shared by the pipeline, the policy, the report, the API and the UI."""

from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


NoteKind = Literal["trust", "provenance", "context", "footer"]
NoteState = Literal["holds", "partly", "breaks", "at_risk", "info"]
Precision = Literal["fp16", "fp32"]
Device = Literal["cuda", "cpu"]
CredentialStatus = Literal["found", "absent", "unsupported", "error"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VideoInfo(_Model):
    """What the decoder read from the file."""

    name: str
    duration_s: float
    width: int
    height: int
    fps: float
    n_frames: int = Field(description="Decoded frames")
    n_kept_frames: int = Field(description="Frames kept after the stride")
    n_clips: int
    frame_stride: int = Field(default=2, description="Every n-th decoded frame is kept")
    rotation: int = Field(
        default=0,
        description="Degrees counter-clockwise to show the video upright. Display only: the models see the frames "
        "as decoded, as in training",
    )


class FaceSummary(_Model):
    """Faces found on the kept frames (MediaPipe FaceDetector)."""

    frames_checked: int
    frames_with_face: int
    frames_with_several: int
    usable_frames: int = Field(description="Frames whose largest face is at least the minimum width")
    median_width_patches: float | None = Field(description="Median width of the largest face, in 24 x 24 grid patches")


class Credentials(_Model):
    """C2PA content credentials read from the file."""

    status: CredentialStatus
    declares_ai: bool | None = None
    summary: str | None = None
    detail: str | None = None


class Prechecks(_Model):
    """Checks run before the models."""

    in_scope: bool
    scope_warnings: list[str] = Field(default_factory=list)
    bits_per_pixel: float | None
    faces: FaceSummary | None
    credentials: Credentials


class ScoreContext(_Model):
    """Where a video score sits in the model's usual ranges (benchmark test videos)."""

    real_rank: float = Field(description="Percent of benchmark real videos that score lower")
    fake_rank: float = Field(description="Percent of benchmark fake videos that score lower")
    real_range: tuple[float, float] = Field(description="5th to 95th percentile on benchmark real videos")
    fake_range: tuple[float, float] = Field(description="5th to 95th percentile on benchmark fake videos")
    high_fake_score: bool = Field(description="Above the real usual range")


class ClipScore(_Model):
    """One 64-frame clip."""

    index: int
    first_frame: int = Field(description="Index of the first kept frame in the original video")
    last_frame: int
    n_valid: int
    fake: float
    real_rank: float | None = Field(default=None, description="Percent of benchmark real videos scoring lower")
    frame_values: list[float] = Field(
        description="Per kept frame: MARS frame contribution, or GenD / DFD-HR frame probability",
    )


class ModelScore(_Model):
    """One model's scores on one video."""

    model: str
    name: str
    fake: float = Field(description="Likelihood of fake: mean fake_prob over clips")
    real: float = Field(description="Likelihood of real: 1 - fake")
    clips: list[ClipScore]
    context: ScoreContext
    frame_values_kind: Literal["contribution", "probability"]
    device: Device
    precision: Precision
    device_note: str | None = Field(default=None, description="Why the model ran on the CPU, if it fell back")
    measures: dict[str, float] = Field(default_factory=dict, description="Values the policy rules read")
    seconds: float


class Agreement(_Model):
    """How the models' scores compare, by usual range (never by raw score)."""

    models: list[str]
    high_fake: list[str]
    disagree: bool
    strength: Literal["consistent", "mixed", "weak"] = Field(
        default="weak",
        description="How consistent the evidence is. Never a verdict",
    )
    lean: Literal["fake", "real"] | None = Field(default=None, description="Direction when consistent")


class Reading(_Model):
    """Which way the detectors lean, by a fixed rule (``explain.policy.assess``).

    It describes the detectors' output against their usual ranges, never what the
    video is, and it never changes a score.
    """

    lean: Literal["fake", "real", "unclear"]
    headline: str = Field(description='For example "Detectors lean fake"')
    reasons: list[str] = Field(default_factory=list, description="One plain sentence")
    action: str = Field(description="The next step for the reader")


class PolicyNote(_Model):
    """One note from the policy rules. Notes explain; they never change a score."""

    kind: NoteKind
    rule: str
    tag: str
    state: NoteState
    model: str | None = None
    title: str
    text: str = Field(description="The headline: the finding in at most 12 plain words")
    detail: str = Field(default="", description="The measured numbers and how the test was done")
    reasons: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    mit: list[str] = Field(default_factory=list)
    collapsed: bool = False


ExperimentKind = Literal["frames", "frames_test", "areas_test", "occlusion", "edit"]


class Experiment(_Model):
    """One test run in a session, recorded by the server with the scores it measured.

    Kinds: ``frames`` (frames hidden by hand, every model re-scored), ``frames_test``
    (the model's own top frames hidden, against random), ``areas_test`` (MARS's top
    areas hidden, against random), ``occlusion`` (an occlusion map computed) and
    ``edit`` (What if edits, every model re-scored).
    """

    kind: ExperimentKind
    clip: int
    model: str | None = Field(default=None, description="Model whose own ranking chose what was hidden")
    hidden: int = Field(default=0, description="Frames or areas hidden")
    frames: list[int] = Field(default_factory=list, description="Frame numbers shown for the hidden frames")
    settings: dict[str, float] = Field(default_factory=dict, description="What if settings")
    before: dict[str, float] = Field(default_factory=dict, description="Clip fake score per model, before")
    after: dict[str, float] = Field(default_factory=dict, description="Clip fake score per model, after")
    random_after: float | None = Field(default=None, description="Score after hiding as many at random")
    flat: bool | None = Field(default=None, description="Occlusion map that barely moves the score")


class Versions(_Model):
    """Everything that decides a result, so the same inputs give the same notes."""

    package: str
    protocol: str
    policy: str
    checkpoints: dict[str, str]


class VideoResult(_Model):
    """The full result of one analysis.

    The large arrays (per-frame patch maps) are kept outside the serialized result; read
    them with :meth:`evidence`.
    """

    video: VideoInfo
    prechecks: Prechecks
    models: dict[str, ModelScore]
    agreement: Agreement
    notes: list[PolicyNote] = Field(default_factory=list)
    reading: Reading | None = None
    jurisdiction: str | None = None
    versions: Versions

    _evidence: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)

    def evidence(self, model: str) -> dict[str, Any]:
        """Return the raw evidence arrays for one model.

        Parameters
        ----------
        model : str
            Model id.

        Returns
        -------
        dict
            For MARS, ``patch_contrib`` per clip as arrays of shape (n_valid, 24, 24).
            Empty when the analysis kept no evidence.
        """
        return self._evidence.get(model, {})

    def set_evidence(self, model: str, arrays: dict[str, Any]) -> None:
        """Attach raw evidence arrays for one model (used by the pipeline).

        Parameters
        ----------
        model : str
            Model id.
        arrays : dict
            Name to array (or list of arrays).
        """
        self._evidence[model] = arrays


def to_float(value: float | np.floating[Any]) -> float:
    """Round a score to three decimals, as the policy reads it.

    Parameters
    ----------
    value : float
        A score.

    Returns
    -------
    float
        The score rounded to three decimals.

    Examples
    --------
    >>> to_float(0.66944)
    0.669
    """
    return round(float(value), 3)
