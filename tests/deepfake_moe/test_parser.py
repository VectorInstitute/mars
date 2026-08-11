"""Tests for eval JSON response parsing."""

import json

from deepfake_moe.eval.parser import parse_response


def _valid_payload() -> dict:
    return {
        "video_fake": True,
        "audio_fake": False,
        "video_fake_prob": 0.8,
        "audio_fake_prob": 0.2,
        "fake_periods": [[1.0, 2.5]],
        "evidence": [
            {
                "modality": "video",
                "type": "lip_sync",
                "period": [1.0, 2.5],
                "note": "mouth lags speech",
            }
        ],
        "reason": "The video stream is manipulated while the audio is authentic.",
    }


def test_parse_clean_json() -> None:
    raw = json.dumps(_valid_payload())
    out = parse_response(raw, has_audio=True)
    assert out.parse_ok
    assert out.video_fake is True
    assert out.audio_fake is False
    assert out.video_fake_prob == 0.8
    assert out.audio_fake_prob == 0.2
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
    payload["audio_fake_prob"] = 0.9
    out = parse_response(json.dumps(payload), has_audio=False)
    assert out.audio_fake is False
    assert out.audio_fake_prob == 0.0


def test_missing_fields_marks_parse_not_ok() -> None:
    raw = json.dumps({"video_fake": True})
    out = parse_response(raw, has_audio=True)
    assert not out.parse_ok


def test_prob_clamped() -> None:
    payload = _valid_payload()
    payload["video_fake_prob"] = 1.7
    payload["audio_fake_prob"] = -0.5
    out = parse_response(json.dumps(payload), has_audio=True)
    assert out.video_fake_prob == 1.0
    assert out.audio_fake_prob == 0.0


def test_invalid_fake_periods_filtered() -> None:
    payload = _valid_payload()
    payload["fake_periods"] = [[1.0, 3.0], [5.0, 4.0], "bad", [0.0, 1.0]]
    out = parse_response(json.dumps(payload), has_audio=True)
    assert out.fake_periods == [[1.0, 3.0], [0.0, 1.0]]


def test_legacy_segments_field_ignored() -> None:
    payload = _valid_payload()
    del payload["fake_periods"]
    payload["segments"] = [[2.0, 4.0]]
    out = parse_response(json.dumps(payload), has_audio=True)
    assert not out.parse_ok
    assert out.fake_periods == []


def test_evidence_optional_and_parsed() -> None:
    out = parse_response(json.dumps(_valid_payload()), has_audio=True)
    assert out.parse_ok
    assert out.evidence and out.evidence[0]["type"] == "lip_sync"

    payload = _valid_payload()
    del payload["evidence"]
    out_without_evidence = parse_response(json.dumps(payload), has_audio=True)
    assert out_without_evidence.parse_ok
    assert out_without_evidence.evidence == []
