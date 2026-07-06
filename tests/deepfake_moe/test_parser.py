"""Tests for eval JSON response parsing."""

import json

from deepfake_moe.eval.parser import parse_response


def _valid_payload() -> dict:
    return {
        "video_fake": True,
        "audio_fake": False,
        "video_confidence": 8,
        "audio_confidence": 3,
        "fake_periods": [[1.0, 2.5]],
        "reason": "lip-sync mismatch",
    }


def test_parse_clean_json() -> None:
    raw = json.dumps(_valid_payload())
    out = parse_response(raw, has_audio=True)
    assert out.parse_ok
    assert out.video_fake is True
    assert out.audio_fake is False
    assert out.video_confidence == 8
    assert out.fake_periods == [[1.0, 2.5]]


def test_parse_markdown_fence() -> None:
    raw = "```json\n" + json.dumps(_valid_payload()) + "\n```"
    out = parse_response(raw, has_audio=True)
    assert out.parse_ok
    assert out.video_fake is True


def test_parse_json_embedded_in_prose() -> None:
    raw = "Here is the result:\n" + json.dumps(_valid_payload()) + "\nDone."
    out = parse_response(raw, has_audio=True)
    assert out.parse_ok


def test_parse_failure_returns_parse_ok_false() -> None:
    out = parse_response("not json at all", has_audio=True, clip_id="c1")
    assert not out.parse_ok
    assert out.reason == "parse_failed"


def test_video_only_forces_audio_fields() -> None:
    payload = _valid_payload()
    payload["audio_fake"] = True
    payload["audio_confidence"] = 9
    out = parse_response(json.dumps(payload), has_audio=False)
    assert out.audio_fake is False
    assert out.audio_confidence == 0


def test_missing_fields_marks_parse_not_ok() -> None:
    raw = json.dumps({"video_fake": True})
    out = parse_response(raw, has_audio=True)
    assert not out.parse_ok


def test_confidence_clamped() -> None:
    payload = _valid_payload()
    payload["video_confidence"] = 99
    payload["audio_confidence"] = -5
    out = parse_response(json.dumps(payload), has_audio=True)
    assert out.video_confidence == 10
    assert out.audio_confidence == 0


def test_invalid_fake_periods_filtered() -> None:
    payload = _valid_payload()
    payload["fake_periods"] = [[1.0, 3.0], [5.0, 4.0], "bad", [0.0, 1.0]]
    out = parse_response(json.dumps(payload), has_audio=True)
    assert out.fake_periods == [[1.0, 3.0], [0.0, 1.0]]


def test_legacy_segments_field_accepted() -> None:
    payload = _valid_payload()
    del payload["fake_periods"]
    payload["segments"] = [[2.0, 4.0]]
    out = parse_response(json.dumps(payload), has_audio=True)
    assert not out.parse_ok
    assert out.fake_periods == [[2.0, 4.0]]
