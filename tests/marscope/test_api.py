"""Tests for the gateway with a stub toolkit (no models)."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from marscope.api import Limits, create_app, sniff_video
from marscope.explain.policy import apply_policy
from marscope.schema import VideoResult


class StubToolkit:
    """Stands in for MARScope: returns a fixed result."""

    device_id = 0

    def __init__(self, result: VideoResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.probes = 0
        self.maps = 0

    def analyze(self, path: Path, **kwargs: Any) -> VideoResult:
        """Return the fixed result with fresh notes."""
        self.calls.append({"path": path, **kwargs})
        kwargs["progress"]({"type": "clip", "model": "mars", "done": 1, "total": 1})
        result = self.result.model_copy(deep=True)
        result.notes = apply_policy(result, kwargs.get("jurisdiction"))
        return result

    def loaded(self) -> list[str]:
        """Return no loaded models."""
        return []

    def rescore(self, _path: Any, result: Any, _clip: int, drop: list[int]) -> dict[str, float]:
        """Return each model's score lowered by the number of runs removed."""
        return {m: round(0.9 - 0.05 * len(drop), 4) for m in result.models}

    def edit_scores(self, _path: Any, result: Any, _clip: int, edits: Any) -> dict[str, float]:
        """Return each model's score lowered as JPEG quality falls."""
        return {m: round(0.9 * edits.jpeg / 100, 4) for m in result.models}

    def quality_sweep(self, _path: Any, result: Any, _clip: int, _edits: Any) -> dict[str, list[float]]:
        """Return a fixed curve per model."""
        return {m: [0.9, 0.8, 0.6, 0.4, 0.3] for m in result.models}

    def sweep(self, _path: Any, _result: Any, model: str, clip: int, **_kwargs: Any) -> dict[str, Any]:
        """Return a fixed curve."""
        return {"model": model, "clip": clip, "order": list(range(16)), "top": [0.9, 0.5], "random": [0.9, 0.88]}

    def occlusion(self, *_args: Any, **_kwargs: Any) -> list[list[float]]:
        """Return a fixed 8 x 8 map with one strong cell."""
        self.maps += 1
        return [[0.2 if (r, c) == (3, 4) else 0.0 for c in range(8)] for r in range(8)]

    def probe(self, *_args: Any, **_kwargs: Any) -> dict[str, Any]:
        """Return a fixed probe result."""
        self.probes += 1
        return {"probe": "remove_top_frames", "count": 6, "before": 0.9, "after": 0.4, "random_after": 0.85}


@pytest.fixture
def client(make_result: Callable[..., VideoResult], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("MARSCOPE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("MARSCOPE_LLM_BASE_URL", raising=False)
    monkeypatch.setattr("marscope.api.serving_summary", lambda _m: {"peak_vram_gb": 2.5, "seconds_per_clip": 0.5})
    monkeypatch.setattr("marscope.api.typical_ranges", lambda _m: {"real_p50": 0.1, "real_p95": 0.24, "fake_p50": 0.29})
    stub = StubToolkit(make_result({"mars": (0.9, True)}))
    return TestClient(create_app(stub, Limits(max_seconds=60)))  # type: ignore[arg-type]


def wait_done(client: TestClient, job_id: str) -> dict[str, Any]:
    for _ in range(100):
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("done", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_sniff_video() -> None:
    assert sniff_video(b"\x00\x00\x00\x20ftypisom") == "mp4"
    assert sniff_video(b"not a video!") is None


def test_pages_and_health(client: TestClient) -> None:
    assert "MARScope" in client.get("/").text
    assert "MARScope" in client.get("/app").text
    assert client.get("/health").json()["status"] == "ok"
    assert client.post("/jobs/abc/feedback", json={}).status_code in (404, 405)
    models = client.get("/models").json()
    assert [m["id"] for m in models] == ["mars", "gend", "dfd-hr"]
    assert models[1]["benchmark"]["avg"] == 88.46
    assert models[0]["typical"]["real_p95"] == 0.24
    assert client.get("/benchmark").json()["sets"][0] == "FF++"
    assert any(r["key"] == "mit-ai-risk" for r in client.get("/references").json())


def test_job_lifecycle(client: TestClient, video: Path) -> None:
    with video.open("rb") as f:
        created = client.post(
            "/jobs", files={"video": ("clip.mp4", f, "video/mp4")}, data={"models": "mars", "jurisdiction": "eu"}
        )
    assert created.status_code == 202
    job_id = created.json()["id"]
    body = wait_done(client, job_id)
    assert body["status"] == "done"
    assert body["result"]["models"]["mars"]["context"]["high_fake_score"] is True
    assert any(r["key"] == "eu-ai-act-art50" for r in body["references"])

    events = [
        json.loads(line[6:])
        for line in client.get(f"/jobs/{job_id}/events").text.splitlines()
        if line.startswith("data: ")
    ]
    assert events[0]["stage"] == "queued" and events[-1]["stage"] == "done"

    report = client.post(f"/jobs/{job_id}/report", json={"format": "json"}).json()
    assert report["summary_source"] == "template"
    pdf = client.post(f"/jobs/{job_id}/report", json={"format": "pdf"})
    assert pdf.content.startswith(b"%PDF")

    probe = client.post(f"/jobs/{job_id}/probe", json={"model": "mars", "clip": 0, "kind": "frames"}).json()
    assert probe["after"] < probe["before"]
    assert client.post(f"/jobs/{job_id}/probe", json={"model": "gend", "clip": 0, "kind": "frames"}).status_code == 404
    same = client.post(f"/jobs/{job_id}/rescore", json={"clip": 0, "drop": []}).json()
    assert same["drop"] == [] and set(same["scores"]) == {"mars"}
    less = client.post(f"/jobs/{job_id}/rescore", json={"clip": 0, "drop": [3, 1, 3]}).json()
    assert less["drop"] == [1, 3] and less["scores"]["mars"] == 0.8
    assert client.post(f"/jobs/{job_id}/rescore", json={"clip": 99, "drop": [1]}).status_code == 404
    assert client.post(f"/jobs/{job_id}/rescore", json={"clip": 0, "drop": [16]}).status_code == 422
    curve = client.post(f"/jobs/{job_id}/sweep", json={"model": "mars", "clip": 0}).json()
    assert curve["top"][-1] < curve["random"][-1]
    assert client.post(f"/jobs/{job_id}/sweep", json={"model": "gend", "clip": 0}).status_code == 404
    edited = client.post(f"/jobs/{job_id}/edit", json={"clip": 0, "jpeg": 50}).json()
    assert edited["scores"]["mars"] == 0.45 and edited["edits"]["jpeg"] == 50
    assert client.post(f"/jobs/{job_id}/edit", json={"clip": 0, "jpeg": 5}).status_code == 422
    chart = client.post(f"/jobs/{job_id}/quality-sweep", json={"clip": 0, "scale": 60}).json()
    assert chart["qualities"][0] == 100 and len(chart["scores"]["mars"]) == len(chart["qualities"])

    renotes = client.post(f"/jobs/{job_id}/notes", json={"jurisdiction": "ca", "language": "fr"}).json()
    assert any(n["rule"] == "context_intimate_ca" for n in renotes["notes"])
    assert client.get(f"/jobs/{job_id}").json()["result"]["jurisdiction"] == "ca"

    assert client.delete(f"/jobs/{job_id}").status_code == 204
    assert client.get(f"/jobs/{job_id}").status_code == 404


def test_rejects_non_video_and_bad_fields(client: TestClient, video: Path) -> None:
    bad = client.post("/jobs", files={"video": ("x.mp4", b"hello world, not a video", "video/mp4")})
    assert bad.status_code == 400
    assert "not an MP4" in bad.json()["error"]
    with video.open("rb") as f:
        assert client.post("/jobs", files={"video": ("clip.mp4", f)}, data={"models": "nope"}).status_code == 400
    with video.open("rb") as f:
        assert client.post("/jobs", files={"video": ("clip.mp4", f)}, data={"jurisdiction": "xx"}).status_code == 400


def test_report_keeps_the_session_tests_and_ask(client: TestClient, video: Path) -> None:
    assert client.get("/health").json()["llm"] is False
    with video.open("rb") as f:
        job_id = client.post("/jobs", files={"video": ("clip.mp4", f, "video/mp4")}, data={"models": "mars"}).json()[
            "id"
        ]
    wait_done(client, job_id)
    client.post(f"/jobs/{job_id}/rescore", json={"clip": 0, "drop": [1]})
    client.post(f"/jobs/{job_id}/rescore", json={"clip": 0, "drop": [1, 3]})  # replaces the first
    client.post(f"/jobs/{job_id}/sweep", json={"model": "mars", "clip": 0})
    client.post(f"/jobs/{job_id}/edit", json={"clip": 0, "jpeg": 100})  # no edit: not a test
    client.post(f"/jobs/{job_id}/edit", json={"clip": 0, "jpeg": 50})
    stub = client.app.state.manager.toolkit  # type: ignore[attr-defined]
    client.post(f"/jobs/{job_id}/probe", json={"model": "mars", "clip": 0, "kind": "patches", "record": False})
    report = client.post(f"/jobs/{job_id}/report", json={"format": "json"}).json()
    assert [r["kind"] for r in report["records"]] == ["frames", "frames_test", "edit"]  # the pre-run is not listed
    client.post(f"/jobs/{job_id}/probe", json={"model": "mars", "clip": 0, "kind": "patches"})  # shown: listed
    assert stub.probes == 1  # computed once, from the cache the second time
    report = client.post(f"/jobs/{job_id}/report", json={"format": "json"}).json()
    assert [r["kind"] for r in report["records"]] == ["frames", "frames_test", "edit", "areas_test"]
    assert report["records"][0]["hidden"] == 2
    assert report["experiments"][0].startswith("Clip 1: with 2 frames hidden")
    assert report["llm_status"] == "off" and "no language model" in report["by"]
    assert "## What you tried" in client.post(f"/jobs/{job_id}/report", json={"format": "markdown"}).text
    frames = client.app.state.manager.jobs[job_id].folder / "frames"  # type: ignore[attr-defined]
    frames.mkdir(exist_ok=True)
    Image.new("RGB", (32, 18), (40, 40, 40)).save(frames / "000004.jpg")
    with_frame = client.post(f"/jobs/{job_id}/report", json={"format": "html", "frame": 4}).text
    assert "data:image/jpeg;base64," in with_frame and "Frame 5 (clip 1)" in with_frame
    assert "data:image" not in client.post(f"/jobs/{job_id}/report", json={"format": "html"}).text
    pdf = client.post(f"/jobs/{job_id}/report", json={"format": "pdf", "frame": 4})
    assert pdf.content.startswith(b"%PDF")
    assert client.post(f"/jobs/{job_id}/report", json={"format": "html", "frame": 7}).status_code == 404
    french = client.post(f"/jobs/{job_id}/report", json={"format": "json", "language": "fr"}).json()
    assert french["experiments"][0].startswith("Extrait 1")

    chip = client.post(f"/jobs/{job_id}/ask", json={"chip": "tests"}).json()
    assert chip["source"] == "fixed" and chip["text"]
    verdict = client.post(f"/jobs/{job_id}/ask", json={"question": "Is this video fake?"}).json()
    assert verdict["refused"] is True and "The final decision is yours" in verdict["text"]
    free = client.post(f"/jobs/{job_id}/ask", json={"question": "Why is the score high?"}).json()
    assert free["text"] == "" and free["llm_status"] == "off"
    assert client.post(f"/jobs/{job_id}/ask", json={}).status_code == 422
    assert client.post(f"/jobs/{job_id}/ask", json={"chip": "tests", "question": "x"}).status_code == 422
    assert client.post(f"/jobs/{job_id}/ask", json={"chip": "nope"}).status_code == 422
    assert client.post(f"/jobs/{job_id}/ask", json={"question": "x" * 501}).status_code == 422


def test_a_gpu_out_of_memory_says_what_to_do(client: TestClient, video: Path) -> None:
    stub = client.app.state.manager.toolkit  # type: ignore[attr-defined]

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("[ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : CUDA failure 2: out of memory")

    stub.analyze = boom
    with video.open("rb") as f:
        job_id = client.post("/jobs", files={"video": ("clip.mp4", f, "video/mp4")}, data={"models": "mars"}).json()[
            "id"
        ]
    body = wait_done(client, job_id)
    assert body["status"] == "failed" and "ran out of memory" in body["error"] and "--sequential" in body["error"]


def test_occlusion_is_cached_and_feeds_the_areas_test(client: TestClient, video: Path) -> None:
    stub = client.app.state.manager.toolkit  # type: ignore[attr-defined]
    with video.open("rb") as f:
        job_id = client.post("/jobs", files={"video": ("clip.mp4", f, "video/mp4")}, data={"models": "mars"}).json()[
            "id"
        ]
    wait_done(client, job_id)
    pre = client.post(f"/jobs/{job_id}/occlusion", json={"model": "mars", "clip": 0, "record": False}).json()
    assert pre["map"][3][4] == 0.2
    client.post(f"/jobs/{job_id}/occlusion", json={"model": "mars", "clip": 0})
    areas = client.post(f"/jobs/{job_id}/probe", json={"model": "mars", "clip": 0, "kind": "areas"})
    assert areas.status_code == 200 and stub.maps == 1  # one map for the pre-run, the map request and the test
    kinds = [r["kind"] for r in client.post(f"/jobs/{job_id}/report", json={"format": "json"}).json()["records"]]
    assert kinds == ["occlusion", "areas_test"]
