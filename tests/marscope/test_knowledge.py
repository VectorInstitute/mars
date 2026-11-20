"""Tests for the knowledge cards and how Ask uses them (no network, no LLM)."""

import re
from collections.abc import Callable
from typing import Any

import pytest

from marscope.explain import ask as ask_module
from marscope.explain.knowledge import MATCH, load_cards, search
from marscope.explain.policy import apply_policy, load_policy
from marscope.explain.report import VERDICT
from marscope.schema import VideoResult


SETTINGS = {"base_url": "https://llm.example.org/v1", "api_key": "k", "model": "test-model"}
MARSCOPE_CARDS = {"marscope", "privacy"}


@pytest.mark.parametrize("card", load_cards(), ids=lambda c: c.id)
def test_every_card_is_short_plain_and_cited(card: Any) -> None:
    known = load_policy()["references"]
    for lang in ("en", "fr"):
        answer = card.text("answer", lang)
        assert card.question.get(lang) and card.answer.get(lang), (card.id, lang)
        assert len(re.split(r"(?<=[.!?])\s+", answer.strip())) <= 3, (card.id, lang)
        assert not re.search(r"\d", answer.replace("C2PA", "")), (card.id, lang)
        assert not VERDICT.search(answer), (card.id, lang)
        assert "relied" not in answer
    assert len(card.text("answer", "en").split()) <= 60, card.id
    assert len(card.keywords) >= 3 and card.source, card.id
    assert all(ref in known for ref in card.references), card.id
    assert card.references or card.id in MARSCOPE_CARDS, card.id


@pytest.mark.parametrize(
    ("question", "card"),
    [
        ("What is a deepfake?", "deepfake"),
        ("C'est quoi un hypertrucage ?", "deepfake"),
        ("Why do the detectors disagree?", "disagree"),
        ("What is C2PA?", "c2pa"),
        ("Can a video be faked with sound?", "voice"),
        ("Is my video uploaded?", "privacy"),
        ("What should our communications team do with this video?", "org_steps"),
        ("what are A1 to A5", "assumptions"),
    ],
)
def test_search_finds_the_card(question: str, card: str) -> None:
    hits = search(question)
    assert hits[0][0].id == card and hits[0][1] >= MATCH


@pytest.mark.parametrize("question", ["What's the weather today?", "what is the capital of France"])
def test_unrelated_questions_find_no_close_card(question: str) -> None:
    hits = search(question)
    assert not hits or hits[0][1] < MATCH


@pytest.fixture
def result(make_result: Callable[..., VideoResult]) -> VideoResult:
    r = make_result({"mars": (0.9, True), "gend": (0.3, False)})
    r.notes = apply_policy(r)
    return r


def stub_llm(answer: str, calls: list[str]) -> Callable[..., tuple[str, str]]:
    def complete(_system: str, user: str, *_args: Any, **_kwargs: Any) -> tuple[str, str]:
        calls.append(user)
        return answer, "used"

    return complete


def test_general_questions_get_the_card_and_never_reach_the_llm(
    result: VideoResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ask_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(ask_module, "complete", stub_llm("Anything.", calls))
    answer = ask_module.ask(result, "What is a deepfake?")
    assert answer.source == "card" and answer.text.startswith("A deepfake is a video")
    assert [r["key"] for r in answer.references] == ["survey"] and not calls
    assert ask_module.ask(result, "Qu’est-ce qu’un hypertrucage ?", "fr").text.startswith("Un hypertrucage")


def test_result_questions_use_the_llm_with_cards_as_background(
    result: VideoResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ask_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(ask_module, "complete", stub_llm("Sharing erases fine detail [survey] [nope].", calls))
    answer = ask_module.ask(result, "Why does sharing a video change the score?")
    assert answer.source == "llm" and answer.text == "Sharing erases fine detail [survey]."
    assert '"background"' in calls[0] and "Social media platforms compress" in calls[0]


def test_without_an_llm(result: VideoResult) -> None:
    assert ask_module.ask(result, "What is C2PA?").source == "card"
    strong = ask_module.ask(result, "Why does sharing a video change the score?")
    assert strong.source == "card"  # about the result, but the sharing card answers it closely
    weak = ask_module.ask(result, "Why is the score high?")
    assert weak.text == "" and weak.llm_status == "off"


def test_deepfake_chip_works_offline(result: VideoResult) -> None:
    answer = ask_module.chip_answer(result, "deepfake")
    assert answer.source == "card" and "deepfake" in answer.text.lower()


def test_about_result() -> None:
    assert ask_module.about_result("Why is this video's score high?")
    assert ask_module.about_result("Pourquoi la probabilité de MARS est-elle basse ?")
    assert not ask_module.about_result("What are Content Credentials?")


def test_a_filtered_llm_answer_falls_back_to_the_card(result: VideoResult, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(ask_module, "llm_settings", lambda: SETTINGS)
    monkeypatch.setattr(ask_module, "complete", stub_llm("This video looks manipulated. Call a lawyer.", calls))
    answer = ask_module.ask(result, "What should our communications team do with this video?")
    assert calls and answer.source == "card" and answer.text.startswith("Do not act on a detector score alone.")
