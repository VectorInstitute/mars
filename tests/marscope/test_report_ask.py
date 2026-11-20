"""Tests for the report with session tests, the LLM fallbacks and Ask (no network)."""

import re
from collections.abc import Callable
from typing import Any

import pytest

from marscope.explain import ask as ask_module
from marscope.explain import report as report_module
from marscope.explain.ask import asks_for_verdict, chip_answer
from marscope.explain.policy import apply_policy
from marscope.explain.report import (
    build_report,
    clean_prose,
    describe_experiment,
    llm_payload,
    to_html,
    to_markdown,
    to_pdf,
)
from marscope.schema import Experiment, VideoResult


Make = Callable[..., VideoResult]
SETTINGS = {"base_url": "https://llm.example.org/v1", "api_key": "k", "model": "test-model"}


def session_tests() -> list[Experiment]:
    return [
        Experiment(kind="frames", clip=0, hidden=2, frames=[9, 25], before={"mars": 0.9}, after={"mars": 0.62}),
        Experiment(
            kind="frames_test",
            clip=0,
            model="mars",
            hidden=4,
            before={"mars": 0.9},
            after={"mars": 0.4},
            random_after=0.86,
        ),
        Experiment(
            kind="areas_test",
            clip=0,
            model="mars",
            hidden=16,
            before={"mars": 0.9},
            after={"mars": 0.7},
            random_after=0.88,
        ),
        Experiment(kind="occlusion", clip=0, model="mars", flat=True),
        Experiment(
            kind="edit",
            clip=0,
            settings={"jpeg": 35.0, "scale": 40.0, "blur": 0.0, "noise": 0.0},
            before={"mars": 0.9},
            after={"mars": 0.3},
        ),
    ]


@pytest.fixture
def result(make_result: Make) -> VideoResult:
    r = make_result({"mars": (0.9, True), "gend": (0.3, False)})
    r.notes = apply_policy(r)
    return r


def fake_llm(answer: str | None, status: str = "used") -> Callable[..., tuple[str | None, str]]:
    def complete(*_args: Any, **_kwargs: Any) -> tuple[str | None, str]:
        return answer, status

    return complete


def test_experiments_in_plain_words(result: VideoResult) -> None:
    lines = [describe_experiment(e, result) for e in session_tests()]
    assert lines[0] == "Clip 1: with 2 frames hidden (frames 9, 25), the scores moved: MARS 90% to 62%."
    assert "hiding MARS’s top 4 frames moved its score from 90% to 40%" in lines[1]
    assert "at random moved it to 86%" in lines[1]
    assert "no area whose hiding changes its score" in lines[3]
    assert lines[4].startswith("Clip 1, edited (JPEG quality 35, 40% resolution)")
    assert "Extrait 1" in describe_experiment(session_tests()[0], result, "fr")
    one = Experiment(kind="frames", clip=1, hidden=1, frames=[70], before={"mars": 0.9}, after={"mars": 0.9})
    assert describe_experiment(one, result).startswith("Clip 2: with frame 70 hidden,")


def test_report_lists_session_tests_in_every_format(result: VideoResult) -> None:
    built = build_report(result, use_llm=False, experiments=session_tests())
    assert built.llm_status == "disabled" and len(built.experiments) == 5
    assert "The 5 tests you ran in the app" in built.summary
    md = to_markdown(built)
    assert "## What you tried" in md and "frames 9, 25" in md
    assert "<h2>What you tried</h2>" in to_html(built)
    assert to_pdf(built).startswith(b"%PDF")
    fr = to_markdown(build_report(result, use_llm=False, experiments=session_tests(), language="fr"))
    assert (
        "## Ce que vous avez essayé" in fr
        and "Très inhabituel pour une vraie vidéo : environ 1 vraie vidéo sur 100" in fr
    )


def test_llm_payload_tests_have_no_raw_scores(result: VideoResult) -> None:
    payload = llm_payload(result, session_tests())
    text = str(payload)
    assert "0.9" not in text and "0.62" not in text
    assert payload["tests_the_user_ran"][1]["effect"] == "large drop"
    assert payload["tests_the_user_ran"][1]["random_effect"] == "little change"


def test_llm_summary_is_used_filtered_and_labelled(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_module, "llm_settings", lambda: SETTINGS)
    prose = "The models read this video differently [survey] [made-up]. The video is fake. MARS gave it 0.9. A4 is worth a look."
    monkeypatch.setattr(report_module, "complete", fake_llm(prose))
    built = build_report(result)
    assert built.summary_source == "llm"
    assert built.summary == "The models read this video differently [survey]. A4 is worth a look."
    assert built.by_line.startswith("Summary written by test-model (llm.example.org)")
    assert "[made-up]" not in to_markdown(built)


@pytest.mark.parametrize(
    ("answer", "status", "expected"),
    [
        (None, "timeout", "timeout"),
        (None, "failed", "failed"),
        ("This video is fake. It scored 0.9.", "used", "filtered"),
    ],
)
def test_llm_failures_fall_back_to_the_template(
    result: VideoResult, monkeypatch: pytest.MonkeyPatch, answer: str | None, status: str, expected: str
) -> None:
    monkeypatch.setattr(report_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(report_module, "complete", fake_llm(answer, status))
    built = build_report(result)
    assert built.summary_source == "template" and built.llm_status == expected
    assert "template" in built.by_line


def test_clean_prose_keeps_test_names_and_drops_french_verdicts() -> None:
    text = "A1 holds for MARS [mars]. La vidéo est fausse. C2PA says nothing here. It scored 87 percent."
    assert clean_prose(text, ["mars"]) == "A1 holds for MARS [mars]. C2PA says nothing here."


@pytest.mark.parametrize("chip", ["disagree", "tests", "certain", "verdict"])
def test_chips_answer_offline_with_attached_references(result: VideoResult, chip: str) -> None:
    allowed = {k for n in result.notes for k in n.references}
    answer = chip_answer(result, chip)
    assert answer.source == "fixed" and answer.text
    assert {r["key"] for r in answer.references} <= allowed
    assert not re.search(r"\b(this|the) video (is|looks|seems)\b", answer.text)
    assert chip_answer(result, chip, "fr").text != answer.text


def test_disagree_chip_compares_by_rank(result: VideoResult) -> None:
    text = chip_answer(result, "disagree").text
    assert text.startswith(
        "They disagree: for MARS, this probability of fake is very unusual for a real video; for GenD,"
    )
    assert len(re.split(r"(?<=[.!?])\s+", text)) <= 3


def test_unknown_chip_is_refused(result: VideoResult) -> None:
    with pytest.raises(ValueError, match="chip must be one of"):
        chip_answer(result, "nope")


def test_verdict_questions_never_reach_the_llm(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("the LLM was called")

    monkeypatch.setattr(ask_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(ask_module, "complete", boom)
    answer = ask_module.ask(result, "So is this video fake or not?")
    assert answer.refused and answer.source == "fixed"
    assert answer.text.startswith("No clear lean: MARS leans fake.") and "The final decision is yours" in answer.text
    assert asks_for_verdict("Est-ce que c'est un hypertrucage ?")
    assert not asks_for_verdict("What does A4 breaks mean?")


def test_free_question_without_llm_says_so(result: VideoResult) -> None:
    answer = ask_module.ask(result, "Why is the score high?")
    assert answer.text == "" and answer.llm_status == "off"


def test_free_question_uses_the_same_filter(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ask_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(
        ask_module, "complete", fake_llm("GenD reads frames one by one [mars] [other]. It gave 30 percent.")
    )
    answer = ask_module.ask(result, "Why does GenD score lower?", experiments=session_tests())
    assert answer.text == "GenD reads frames one by one [mars]."
    assert [r["key"] for r in answer.references] == ["mars"] and answer.llm["model"] == "test-model"


def test_number_words_and_wrong_premises(result: VideoResult) -> None:
    text = "Its score is forty-four percent, in the real range. One score is not proof [survey]. Both models ran."
    assert clean_prose(text, ["survey"]) == "One score is not proof [survey]. Both models ran."
    assert (
        clean_prose("Le score est de quarante pour cent. Les deux modèles divergent.", [])
        == "Les deux modèles divergent."
    )
    payload = llm_payload(result)
    mars = payload["models"][0]
    assert mars["compared_with_real_videos"] == "very unusual for a real video"
    assert payload["models"][1]["compared_with_real_videos"] == "typical for a real video"
    assert all("%" not in n["note"] for n in payload["notes"])


def test_llm_answers_are_capped(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    long = "The models agree. The scores sit high. One score is not proof. This is extra. And more."
    monkeypatch.setattr(report_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(report_module, "complete", fake_llm(long))
    assert build_report(result).summary == "The models agree. The scores sit high. One score is not proof."
    monkeypatch.setattr(ask_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(ask_module, "complete", fake_llm(long))
    assert ask_module.ask(result, "Why?").text.count(".") == 3


@pytest.mark.parametrize(
    "sentence",
    [
        "This indicates a high probability of fake content.",
        "This suggests the content may be synthetic.",
        "The clip looks manipulated.",
        "La vidéo est probablement truquée.",
        "You can trust these findings as strong evidence.",
        "This confirms the result.",
        "It is likely a deepfake.",
    ],
)
def test_filter_drops_implied_verdicts(sentence: str) -> None:
    assert clean_prose(f"One score is not proof [survey]. {sentence}", ["survey"]) == "One score is not proof [survey]."


def test_filter_keeps_plain_descriptions() -> None:
    text = (
        "Every model gives a probability of fake that is very unusual for a real video. "
        "The score drops when the video is shared many times. One score is not proof."
    )
    assert clean_prose(text, []) == text


def test_summary_that_loses_its_first_sentence_falls_back(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(
        report_module, "complete", fake_llm("The video looks manipulated. One score is not proof. Check the source.")
    )
    built = build_report(result)
    assert built.summary_source == "template" and built.llm_status == "filtered"
    monkeypatch.setattr(report_module, "complete", fake_llm("No tests were run."))
    assert build_report(result).summary_source == "template"  # one sentence is not a summary


def test_summary_drops_test_mentions_when_none_were_run(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(report_module, "llm_settings", lambda: SETTINGS)
    prose = "MARS gives a probability of fake that is very unusual for a real video. No user tests were run to verify this finding. A single score is not proof."
    monkeypatch.setattr(report_module, "complete", fake_llm(prose))
    built = build_report(result)
    assert built.summary_source == "llm" and "tests" not in built.summary
    assert built.summary.endswith("A single score is not proof.")
    with_tests = build_report(result, experiments=session_tests())
    assert "No user tests" in with_tests.summary  # only dropped when no tests ran
