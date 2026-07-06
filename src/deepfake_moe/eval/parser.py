"""Parse and validate model JSON output into ``PredictionOutput``.

Extraction order: direct JSON parse, markdown fence, then regex object search.
For video-only clips, audio fields are forced to null values.
"""

from __future__ import annotations

import json
import re

from deepfake_moe.models.base import PredictionOutput
from deepfake_moe.utils.logging_utils import get_logger


logger = get_logger(__name__)

_REQUIRED_FIELDS = frozenset(
    {"video_fake", "audio_fake", "video_confidence", "audio_confidence", "fake_periods", "reason"}
)


def parse_response(
    raw: str,
    *,
    has_audio: bool,
    clip_id: str = "",
) -> PredictionOutput:
    """Parse raw model text into a ``PredictionOutput``.

    Parameters
    ----------
    raw :
        Verbatim text returned by the model.
    has_audio :
        When ``False``, audio fields are forced to null values.
    clip_id :
        Used only for log messages.

    Returns
    -------
    PredictionOutput
        ``parse_ok=False`` when JSON could not be extracted.
    """
    data = _extract_json(raw)

    if data is None:
        logger.warning("clip=%s parse_failed raw=%r", clip_id, raw[:300])
        return PredictionOutput(
            video_fake=False,
            audio_fake=False,
            video_confidence=0,
            audio_confidence=0,
            fake_periods=[],
            reason="parse_failed",
            raw_response=raw,
            parse_ok=False,
        )

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        logger.warning("clip=%s missing_fields=%s", clip_id, sorted(missing))

    video_fake = bool(data.get("video_fake", False))
    audio_fake = bool(data.get("audio_fake", False))
    vc = _clamp(data.get("video_confidence", 0))
    ac = _clamp(data.get("audio_confidence", 0))
    fake_periods = _parse_fake_periods(data.get("fake_periods", data.get("segments", [])))
    reason = str(data.get("reason", ""))

    if not has_audio:
        audio_fake = False
        ac = 0

    return PredictionOutput(
        video_fake=video_fake,
        audio_fake=audio_fake,
        video_confidence=vc,
        audio_confidence=ac,
        fake_periods=fake_periods,
        reason=reason,
        raw_response=raw,
        parse_ok=not bool(missing),
    )


def _extract_json(text: str) -> dict | None:
    stripped = text.strip()
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    return None


def _clamp(value: object) -> int:
    try:
        if isinstance(value, (int, float, str)):
            return max(0, min(10, int(value)))
    except (TypeError, ValueError):
        pass
    return 0


def _parse_fake_periods(raw: object) -> list[list[float]]:
    if not isinstance(raw, list):
        return []
    out: list[list[float]] = []
    for seg in raw:
        if isinstance(seg, (list, tuple)) and len(seg) == 2:
            try:
                t0, t1 = float(seg[0]), float(seg[1])
                if t0 < t1:
                    out.append([t0, t1])
            except (TypeError, ValueError):
                continue
    return out
