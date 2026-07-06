"""Tests for per-head detection metrics."""

from deepfake_moe.eval.metrics.detection import compute_detection_metrics


def _record(
    *,
    label_video: int = 0,
    label_audio: int = 0,
    video_fake: bool = False,
    audio_fake: bool = False,
    video_confidence: int = 5,
    audio_confidence: int = 5,
    has_audio: bool = True,
    parse_ok: bool = True,
) -> dict:
    return {
        "label_video": label_video,
        "label_audio": label_audio,
        "video_fake": video_fake,
        "audio_fake": audio_fake,
        "video_confidence": video_confidence,
        "audio_confidence": audio_confidence,
        "has_audio": has_audio,
        "parse_ok": parse_ok,
    }


def test_video_head_perfect_accuracy() -> None:
    records = [
        _record(label_video=0, video_fake=False, video_confidence=9),
        _record(label_video=1, video_fake=True, video_confidence=9),
    ]
    m = compute_detection_metrics(records, "video")
    assert m["n_evaluated"] == 2
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0


def test_audio_head_excludes_no_audio_clips() -> None:
    records = [
        _record(label_audio=1, audio_fake=True, has_audio=True),
        _record(label_audio=0, audio_fake=False, has_audio=False),
    ]
    m = compute_detection_metrics(records, "audio")
    assert m["n_evaluated"] == 1


def test_skips_parse_failures() -> None:
    records = [
        _record(parse_ok=False),
        _record(label_video=1, video_fake=True),
    ]
    m = compute_detection_metrics(records, "video")
    assert m["n_evaluated"] == 1
    assert m["n_total"] == 2


def test_invalid_head_raises() -> None:
    try:
        compute_detection_metrics([], "invalid")
    except ValueError as exc:
        assert "video" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_empty_valid_set_returns_note() -> None:
    m = compute_detection_metrics([_record(parse_ok=False)], "video")
    assert m["n_evaluated"] == 0
    assert "note" in m
