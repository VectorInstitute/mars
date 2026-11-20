"""Every policy rule both fires and stays silent. Rules never change a score."""

from collections.abc import Callable

import pytest

from marscope.explain.policy import apply_policy, context_applies, load_policy, references
from marscope.schema import Credentials, FaceSummary, PolicyNote, Reading, VideoResult


Make = Callable[..., VideoResult]


def full(note: PolicyNote) -> str:
    """Headline and detail together, as the reader sees them."""
    return f"{note.text} {note.detail}"


def rules(result: VideoResult, **kwargs: object) -> list[str]:
    return [n.rule for n in apply_policy(result, **kwargs)]  # type: ignore[arg-type]


def test_footer_always_and_last(make_result: Make) -> None:
    notes = apply_policy(make_result())
    assert notes[-1].rule == "footer"
    assert "model outputs" in notes[-1].text and "leaves the decision to people" in notes[-1].detail


def test_rules_never_change_scores(make_result: Make) -> None:
    result = make_result({"mars": (0.9, True), "gend": (0.4, False)}, measures={"mars": {**A1_HOLDS, **A5_BREAKS}})
    before = result.model_dump()
    apply_policy(result, jurisdiction="eu")
    assert result.model_dump() == before


# The fixture's benchmark real range ends at 0.241.
A1_HOLDS = {"a1_before": 0.66, "a1_after": 0.65, "a1_inside": 0.51}
A1_PARTLY = {"a1_before": 0.66, "a1_after": 0.50, "a1_inside": 0.51}
A1_BREAKS = {"a1_before": 0.66, "a1_after": 0.20, "a1_inside": 0.51}
A1_LOW = {"a1_before": 0.10, "a1_after": 0.05, "a1_inside": 0.08}
A5_BREAKS = {"a5_before": 0.66, "a5_after": 0.10}


@pytest.mark.parametrize("model", ["mars", "gend", "dfd-hr"])
@pytest.mark.parametrize(
    ("measures", "expected"),
    [(A1_HOLDS, "a1_holds"), (A1_PARTLY, "a1_partly"), (A1_BREAKS, "a1_breaks"), (A1_LOW, "a1_na")],
)
def test_a1_answers_for_every_model(make_result: Make, model: str, measures: dict[str, float], expected: str) -> None:
    notes = apply_policy(make_result({model: (0.9, True)}, measures={model: measures}))
    a1 = [n for n in notes if n.tag == "A1"]
    assert [(n.rule, n.model) for n in a1] == [(expected, model)]
    assert f"{round(measures['a1_before'] * 100)}%" in a1[0].detail


def test_a1_and_a5_silent_without_a_test(make_result: Make) -> None:
    assert not any(r.startswith(("a1", "a5")) for r in rules(make_result({"gend": (0.9, True)})))


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [(0.66, 0.64, "a5_holds"), (0.66, 0.50, "a5_partly"), (0.66, 0.10, "a5_breaks"), (0.1, 0.1, "a5_na")],
)
@pytest.mark.parametrize("model", ["mars", "dfd-hr"])
def test_a5(make_result: Make, model: str, before: float, after: float, expected: str) -> None:
    result = make_result({model: (0.9, True)}, measures={model: {"a5_before": before, "a5_after": after}})
    assert [(n.rule, n.model) for n in apply_policy(result) if n.tag == "A5"] == [(expected, model)]


def test_a2_from_out_of_domain_results(make_result: Make) -> None:
    notes = [n for n in apply_policy(make_result({"mars": (0.1, False), "gend": (0.9, True)})) if n.tag == "A2"]
    assert [(n.rule, n.model) for n in notes] == [("a2_info", "mars"), ("a2_info", "gend")]
    assert "93.4" in notes[0].text and "75.8" in notes[0].text  # the headline keeps these two numbers


def test_no_verdict_words_in_trust_notes(make_result: Make) -> None:
    result = make_result({"mars": (0.9, True)}, measures={"mars": {**A1_BREAKS, **A5_BREAKS}})
    for note in apply_policy(result):
        if note.kind == "trust":
            text = full(note)
            assert "relied" not in text and "is fake" not in text and "is real" not in text


def test_one_note_per_model_in_model_order(make_result: Make) -> None:
    result = make_result(
        {"mars": (0.9, True), "gend": (0.9, True), "dfd-hr": (0.1, False)},
        measures={"mars": A1_HOLDS, "gend": A1_BREAKS, "dfd-hr": A1_LOW},
    )
    a1 = [(n.rule, n.model) for n in apply_policy(result) if n.tag == "A1"]
    assert a1 == [("a1_holds", "mars"), ("a1_breaks", "gend"), ("a1_na", "dfd-hr")]


def test_a3_merges_reasons_into_one_note(make_result: Make) -> None:
    faces = FaceSummary(
        frames_checked=75, frames_with_face=0, frames_with_several=0, usable_frames=0, median_width_patches=None
    )
    result = make_result({"mars": (0.9, True), "gend": (0.3, False)}, faces=faces, scope=["short"])
    notes = [n for n in apply_policy(result) if n.rule == "a3_at_risk"]
    assert len(notes) == 1
    assert notes[0].reasons == ["short", "no_face", "disagree"]
    assert "MARS gave a high fake score, GenD did not" in notes[0].detail


def test_a3_face_checks(make_result: Make) -> None:
    partly = FaceSummary(
        frames_checked=100, frames_with_face=30, frames_with_several=30, usable_frames=0, median_width_patches=3.0
    )
    note = next(n for n in apply_policy(make_result(faces=partly)) if n.rule == "a3_at_risk")
    assert note.reasons == ["face_partly", "several_faces", "small_face"]
    assert "30%" in note.detail


def test_a3_silent_on_a_clean_video(make_result: Make) -> None:
    assert "a3_at_risk" not in rules(make_result({"mars": (0.9, True), "gend": (0.9, True)}))


@pytest.mark.parametrize(("after", "expected"), [(0.64, "a4_holds"), (0.5, "a4_partly"), (0.2, "a4_breaks")])
def test_a4_shared_many_times(make_result: Make, after: float, expected: str) -> None:
    result = make_result({"gend": (0.9, True)}, measures={"gend": {"a4_before": 0.66, "a4_after": after}})
    notes = [n for n in apply_policy(result) if n.tag == "A4"]
    assert [(n.rule, n.model) for n in notes] == [(expected, "gend")]
    assert "66%" in notes[0].detail


def test_a4_bitrate_hint(make_result: Make) -> None:
    assert "a4_at_risk" in rules(make_result(bits_per_pixel=0.01))
    assert "a4_at_risk" not in rules(make_result(bits_per_pixel=0.2))
    assert "a4_at_risk" not in rules(make_result(bits_per_pixel=None))


def test_trust_notes_in_assumption_order(make_result: Make) -> None:
    result = make_result({"mars": (0.9, True)}, measures={"mars": {**A1_HOLDS, **A5_BREAKS}}, bits_per_pixel=0.01)
    trust = [n.rule for n in apply_policy(result) if n.kind == "trust"]
    assert trust == ["a1_holds", "a2_info", "a4_at_risk", "a5_breaks"]


def test_provenance(make_result: Make) -> None:
    assert "credentials_absent" in rules(make_result())
    assert "credentials_unsupported" in rules(make_result(credentials=Credentials(status="unsupported")))
    declared = Credentials(status="found", declares_ai=True, summary="made with Gen")
    assert {"credentials_found", "credentials_differ"} <= set(rules(make_result(credentials=declared)))
    assert "credentials_differ" not in rules(make_result({"mars": (0.9, True)}, credentials=declared))
    assert not [r for r in rules(make_result(credentials=Credentials(status="error"))) if r.startswith("credentials")]


def test_context_needs_two_high_scores_of_several(make_result: Make) -> None:
    assert not context_applies(make_result({"mars": (0.9, True), "gend": (0.3, False), "dfd-hr": (0.2, False)}))
    assert context_applies(make_result({"mars": (0.9, True), "gend": (0.9, True), "dfd-hr": (0.2, False)}))
    assert context_applies(make_result({"mars": (0.9, True)}))
    assert not context_applies(make_result({"mars": (0.1, False)}))
    assert context_applies(make_result(credentials=Credentials(status="found", declares_ai=True)))


def test_context_by_jurisdiction_and_collapsed(make_result: Make) -> None:
    result = make_result({"mars": (0.9, True)})
    assert [r for r in rules(result) if r.startswith("context")] == ["context_risks"]
    eu = [n for n in apply_policy(result, jurisdiction="eu") if n.kind == "context"]
    assert {n.rule for n in eu} == {"context_labelling", "context_risks"}
    assert all(n.collapsed for n in eu)
    assert {n.rule for n in apply_policy(result, jurisdiction="ca") if n.kind == "context"} == {
        "context_intimate_ca",
        "context_risks",
    }
    with pytest.raises(ValueError):
        apply_policy(result, jurisdiction="xx")


def test_context_texts_are_conditional() -> None:
    for note in load_policy()["notes"].values():
        if note["kind"] == "context" and note["tag"] != "Deepfake risks":
            assert note["detail"]["en"].startswith("If ")


def test_french_notes(make_result: Make) -> None:
    notes = apply_policy(make_result(bits_per_pixel=0.01), language="fr")
    assert any("bits par pixel" in n.detail for n in notes)


def test_every_note_has_both_languages_and_known_references() -> None:
    policy = load_policy()
    for key, note in policy["notes"].items():
        assert note["text"]["en"] and note["text"]["fr"] and note["detail"]["en"] and note["detail"]["fr"], key
        assert len(note["text"]["en"].split()) <= 12, key  # a headline, read at a glance
        for ref in note.get("references", []):
            assert ref in policy["references"], (key, ref)
    for reason in policy["reasons"].values():
        assert reason["en"] and reason["fr"]


def test_no_verdict_words_in_notes() -> None:
    for note in load_policy()["notes"].values():
        text = (note["text"]["en"] + " " + note["detail"]["en"]).lower()
        assert "this video is fake" not in text and "is a deepfake" not in text and "face swap" not in text


def test_references_lookup_dedupes() -> None:
    out = references(["mars", "survey", "mars", "unknown"])
    assert [r["key"] for r in out] == ["mars", "survey"]


def _reading(make_result: Make, scores: dict[str, tuple[float, bool]], **kwargs: object) -> Reading:
    from marscope.explain.policy import assess  # noqa: PLC0415

    result = make_result(scores, **kwargs)
    result.notes = apply_policy(result)
    return assess(result)


def test_reading_leans_fake_when_every_model_scores_high(make_result: Make) -> None:
    r = _reading(make_result, {"mars": (0.9, True), "gend": (0.9, True)})
    assert (r.lean, r.headline) == ("fake", "Detectors lean fake")
    assert r.reasons == ["Every model gives a probability of fake well above the usual range for real videos."]
    assert r.action == "Check the source before acting."


def test_reading_names_the_single_model(make_result: Make) -> None:
    r = _reading(make_result, {"mars": (0.9, True)})
    assert (r.lean, r.headline) == ("fake", "MARS leans fake")
    assert not any("agree" in x for x in r.reasons)


def test_reading_leans_real_when_every_model_scores_low(make_result: Make) -> None:
    r = _reading(make_result, {"mars": (0.1, False), "gend": (0.1, False)})
    assert (r.lean, r.headline) == ("real", "Detectors lean real")


def test_reading_is_unclear_when_models_disagree(make_result: Make) -> None:
    r = _reading(make_result, {"mars": (0.9, True), "gend": (0.3, False)})
    assert (r.lean, r.headline) == ("unclear", "No clear lean")
    assert r.reasons == ["MARS leans fake. GenD leans real."]
    r = _reading(make_result, {"mars": (0.9, True), "gend": (0.3, False), "dfd-hr": (0.9, True)})
    assert r.reasons == ["MARS and DFD-HR lean fake. GenD leans real."]


def test_reading_in_french(make_result: Make) -> None:
    from marscope.explain.policy import assess  # noqa: PLC0415

    result = make_result({"mars": (0.9, True)})
    result.notes = apply_policy(result, language="fr")
    assert assess(result, "fr").headline == "MARS penche vers le faux"
