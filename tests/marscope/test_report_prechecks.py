"""Tests for pre-checks, the report and the CLI summary."""

from collections.abc import Callable
from pathlib import Path

import pytest

from marscope.cli import build_parser, format_summary
from marscope.explain.policy import apply_policy
from marscope.explain.report import (
    build_report,
    clean_prose,
    llm_payload,
    template_summary,
    to_html,
    to_markdown,
    to_pdf,
)
from marscope.schema import VideoResult
from marscope.video.prechecks import bits_per_pixel, read_credentials, scope_warnings, summarize_credentials


Make = Callable[..., VideoResult]


def test_bits_per_pixel() -> None:
    assert bits_per_pixel(100, 10, 10, 1) == 8.0
    assert bits_per_pixel(100, 0, 10, 10) is None


def test_scope_warnings() -> None:
    assert scope_warnings(200, 64, 1280, 720) == []
    assert scope_warnings(10, 64, 2000, 400) == ["short", "wide"]


def test_credentials_unsupported_for_webm(tmp_path: object) -> None:
    assert read_credentials("clip.webm").status == "unsupported"


def test_credentials_absent_on_a_plain_mp4(video: object) -> None:
    assert read_credentials(str(video)).status in ("absent", "error")


def test_summarize_credentials_without_ai() -> None:
    store = {
        "active_manifest": "m",
        "manifests": {"m": {"claim_generator": "Camera/2", "signature_info": {"issuer": "Org"}}},
    }
    cred = summarize_credentials(store)
    assert cred.declares_ai is False
    assert cred.summary == "made with Camera/2; signed by Org"


def test_clean_prose_drops_numbers_verdicts_and_foreign_citations() -> None:
    text = "<think>x</think>MARS leans on the outline [mars] [nope]. The video is likely fake. It seems genuine. Scores near 0.9 vary. Notes explain limits [survey]."
    assert clean_prose(text, ["mars", "survey"]) == "MARS leans on the outline [mars]. Notes explain limits [survey]."


def test_template_summary_wording(make_result: Make) -> None:
    text = template_summary(make_result({"mars": (0.9, True), "gend": (0.3, False)}))
    assert "MARS gave a probability of fake that is very unusual for a real video; the others did not." in text
    assert "benchmark" not in text
    assert "fake" not in text.lower().replace("probability of fake", "")
    assert "What you tried" in template_summary(make_result(), n_experiments=2)
    assert "a été appliqué" in template_summary(make_result(), "fr")


def test_report_without_llm_and_exports(make_result: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MARSCOPE_LLM_BASE_URL", raising=False)
    result = make_result(
        {"mars": (0.9, True)}, measures={"mars": {"a1_before": 0.9, "a1_after": 0.5, "a1_inside": 0.4}}
    )
    result.notes = apply_policy(result)
    report = build_report(result)
    assert report.summary_source == "template"
    assert {r["key"] for r in report.refs} >= {"mars", "survey"}
    md = to_markdown(report)
    assert "Very unusual for a real video: about 1 in 100 real videos score this high on MARS" in md
    assert "## Disclaimer" in md
    assert "no language model is set up" in md
    assert "<table>" in to_html(report)
    assert to_pdf(report).startswith(b"%PDF")


def test_llm_payload_has_no_raw_scores(make_result: Make) -> None:
    result = make_result({"mars": (0.9, True)})
    result.notes = apply_policy(result)
    payload = llm_payload(result)
    assert "0.9" not in str(payload["models"])


def test_cli_summary_and_parser(make_result: Make) -> None:
    result = make_result({"mars": (0.9, True)})
    result.notes = apply_policy(result)
    text = format_summary(result.model_dump(mode="json"))
    assert "probability of fake 90%" in text
    assert "Very unusual for a real video: about 1 in 100 real videos score this high on MARS" in text
    assert "Scores are model outputs and can be wrong" in text
    args = build_parser().parse_args(["--video", "x.mp4", "--models", "mars", "--json"])
    assert args.json and args.models == "mars" and args.precision == "auto"
    assert build_parser().parse_args(["--serve", "--precision", "fp32"]).precision == "fp32"
    from marscope.models.registry import parse_models  # noqa: PLC0415

    assert build_parser().parse_args(["--serve"]).models is None
    assert parse_models(None) == ["mars"] and parse_models("") == ["mars"]
    assert parse_models("all") == ["mars", "gend", "dfd-hr"]


def test_cli_reports_an_unreadable_video(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from marscope.cli import main  # noqa: PLC0415

    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00not a video" * 100)
    assert main(["--video", str(junk), "--no-faces"]) == 2
    assert "could not read junk.mp4" in capsys.readouterr().err


def test_serve_banner_links_to_the_app() -> None:
    from marscope.cli import serve_banner  # noqa: PLC0415

    assert serve_banner("0.0.0.0", 8000).splitlines()[0].startswith("MARScope at http://localhost:8000/app")  # noqa: S104
    assert "ssh -N -L 8010:" in serve_banner("0.0.0.0", 8000)  # noqa: S104
    assert len(serve_banner("127.0.0.1", 8000).splitlines()) == 1


def test_cli_warns_past_the_web_limits() -> None:
    from marscope.cli import size_warnings  # noqa: PLC0415
    from marscope.video.preprocess import VideoMeta  # noqa: PLC0415

    long = VideoMeta(duration_s=130.0, width=3840, height=2160, fps=30.0, size_bytes=1)
    warnings = size_warnings(long, 3)
    assert (
        len(warnings) == 2 and "about 31 clips for each of 3 model(s)" in warnings[0] and "above 1080p" in warnings[1]
    )
    assert size_warnings(VideoMeta(duration_s=10.0, width=1280, height=720, fps=30.0, size_bytes=1), 1) == []


def test_damaged_files_get_a_plain_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from marscope.cli import main  # noqa: PLC0415

    junk = tmp_path / "half.mp4"
    junk.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 400)
    assert main(["--video", str(junk), "--no-faces"]) == 2
    assert "damaged or incomplete" in capsys.readouterr().err


def test_env_file_is_found_where_marscope_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import dotenv  # noqa: PLC0415

    from marscope.explain.report import llm_settings  # noqa: PLC0415

    monkeypatch.setattr("dotenv.load_dotenv", dotenv.main.load_dotenv)  # the real loader, for this test only
    (tmp_path / ".env").write_text("MARSCOPE_LLM_BASE_URL=https://llm.example.org/v1\nMARSCOPE_LLM_API_KEY=k\n")
    monkeypatch.chdir(tmp_path)
    settings = llm_settings()
    assert settings is not None and settings["base_url"] == "https://llm.example.org/v1"
    for key in ("MARSCOPE_LLM_BASE_URL", "MARSCOPE_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)
