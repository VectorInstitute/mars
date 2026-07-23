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

# parse_ok gates whether a clip counts toward detection metrics. The locked
# schema always includes fake_periods (as [] for a real clip, never omitted —
# see train/common/targets.py's build_target), so its absence signals a
# schema-deviating response (e.g. a legacy `segments` field) rather than a
# real clip with no periods, and must fail the parse like the other required
# fields. ``reason`` is still recovered-when-present (incl. via truncated-JSON
# repair) and defaults to "" otherwise, since it doesn't gate correctness.
_REQUIRED_FIELDS = frozenset({"video_fake", "audio_fake", "video_fake_prob", "audio_fake_prob", "fake_periods"})


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
            video_fake_prob=0.0,
            audio_fake_prob=0.0,
            fake_periods=[],
            evidence=[],
            reason="parse_failed",
            raw_response=raw,
            parse_ok=False,
        )

    missing = _REQUIRED_FIELDS - set(data.keys())
    if missing:
        logger.warning("clip=%s missing_fields=%s", clip_id, sorted(missing))

    video_fake = bool(data.get("video_fake", False))
    audio_fake = bool(data.get("audio_fake", False))
    v_prob = _parse_prob(data.get("video_fake_prob"), fallback=video_fake)
    a_prob = _parse_prob(data.get("audio_fake_prob"), fallback=audio_fake)
    fake_periods = _parse_fake_periods(data.get("fake_periods", []))
    evidence = _parse_evidence(data.get("evidence", []))
    reason = str(data.get("reason", ""))

    if not has_audio:
        audio_fake = False
        a_prob = 0.0

    return PredictionOutput(
        video_fake=video_fake,
        audio_fake=audio_fake,
        video_fake_prob=v_prob,
        audio_fake_prob=a_prob,
        fake_periods=fake_periods,
        evidence=evidence,
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

    # Last resort: repair a generation truncated mid-object (token budget hit).
    # Recovers every field emitted before the cut — crucially ``fake_periods``,
    # which the schema emits before ``evidence``/``reason`` and so survives most
    # truncations.
    return _repair_truncated_json(text)


def _repair_truncated_json(text: str) -> dict | None:
    """Close a JSON object truncated mid-generation and parse the largest valid prefix.

    Fields the model never reached are simply absent (they default
    downstream); everything it did emit before the cut is preserved.
    """
    start = text.find("{")
    if start == -1:
        return None
    s = text[start:]
    st_in, st_esc, st_stack = _scan_json_prefix_state(s)
    return _largest_valid_json_prefix(s, st_in, st_esc, st_stack)


def _scan_json_prefix_state(s: str) -> tuple[list[bool], list[bool], list[tuple[str, ...]]]:
    """Scan ``s`` once and record its per-prefix-length parse state.

    Recorded after every prefix length: whether we're inside a string (and
    mid-escape), and the open ``{``/``[`` stack.
    """
    in_str = False
    esc = False
    stack: list[str] = []
    st_in = [False]
    st_esc = [False]
    st_stack: list[tuple[str, ...]] = [()]
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()
        st_in.append(in_str)
        st_esc.append(esc)
        st_stack.append(tuple(stack))
    return st_in, st_esc, st_stack


def _largest_valid_json_prefix(
    s: str,
    st_in: list[bool],
    st_esc: list[bool],
    st_stack: list[tuple[str, ...]],
) -> dict | None:
    """Return the first valid dict parsed from the longest closeable prefix.

    Tries the longest prefix first: closes a dangling string, drops a
    trailing comma/colon, and appends the missing closers before parsing.
    """
    closers = {"{": "}", "[": "]"}
    for end in range(len(s), 0, -1):
        core = _close_dangling_string(s[:end], in_str=st_in[end], esc=st_esc[end])
        if core is None:
            continue
        candidate = core + "".join(closers[b] for b in reversed(st_stack[end]))
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _close_dangling_string(core: str, *, in_str: bool, esc: bool) -> str | None:
    """Close an open string, or strip a trailing comma/colon.

    Returns ``None`` when the prefix ends mid-token (e.g. right after a
    comma or colon) and can't be closed into valid JSON.
    """
    if in_str:
        return core[:-1] + '"' if esc else core + '"'
    core = core.rstrip()
    while core and core[-1] == ",":
        core = core[:-1].rstrip()
    if not core or core[-1] in ":,":
        return None
    return core


def _parse_prob(value: object, *, fallback: bool) -> float:
    """Coerce a written probability into ``[0.0, 1.0]``."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    if isinstance(value, str):
        try:
            return max(0.0, min(1.0, float(value)))
        except ValueError:
            pass
    return 1.0 if fallback else 0.0


def _parse_evidence(raw: object) -> list[dict]:
    """Keep well-formed evidence items without failing the whole parse."""
    if not isinstance(raw, list):
        return []
    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        clean: dict = {}
        for key in ("modality", "type", "note"):
            if key in item and item[key] is not None:
                clean[key] = str(item[key])
        period = item.get("period")
        if isinstance(period, (list, tuple)) and len(period) == 2:
            try:
                t0, t1 = float(period[0]), float(period[1])
                clean["period"] = [t0, t1]
            except (TypeError, ValueError):
                pass
        out.append(clean)
    return out


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
