"""Evidence strength: how consistent the models are, by usual range. Never a verdict."""

from marscope.pipeline import evidence_strength
from tests.conftest import model_score


def scores(**ranks: float) -> dict:
    out = {}
    for model, rank in ranks.items():
        s = model_score(model.replace("_", "-"), 0.5, rank > 95)
        s.context.real_rank = rank
        out[model.replace("_", "-")] = s
    return out


def test_mixed_when_models_disagree() -> None:
    s = scores(mars=99, gend=40)
    a = evidence_strength(list(s), ["mars"], s)
    assert (a.strength, a.lean, a.disagree) == ("mixed", None, True)


def test_consistent_fake_when_all_high() -> None:
    s = scores(mars=99, gend=98)
    assert (
        evidence_strength(list(s), ["mars", "gend"], s).strength,
        evidence_strength(list(s), ["mars", "gend"], s).lean,
    ) == ("consistent", "fake")


def test_consistent_real_only_when_all_low() -> None:
    s = scores(mars=30, dfd_hr=50)
    a = evidence_strength(list(s), [], s)
    assert (a.strength, a.lean) == ("consistent", "real")
    s = scores(mars=30, dfd_hr=90)
    assert evidence_strength(list(s), [], s).strength == "weak"


def test_one_model() -> None:
    s = scores(mars=99)
    assert evidence_strength(["mars"], ["mars"], s).lean == "fake"
