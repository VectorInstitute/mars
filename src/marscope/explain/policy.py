"""Policy rules: turn a result into fixed, cited notes. Rules never change a score.

Each rule is a small function that reads the result and returns firings (which note,
which reasons). Texts, references, MIT tags and cut-offs live in ``policy.yaml``. Both
are versioned together.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache
from importlib.resources import files
from typing import Any

import yaml

from marscope.analysis.ranges import REAL_LEAN_RANK
from marscope.schema import ModelScore, PolicyNote, Reading, VideoResult


JURISDICTIONS = ("eu", "ca", "us", "all")
KIND_ORDER = {"trust": 0, "provenance": 1, "context": 2, "footer": 3}
STATE_ORDER = {"at_risk": 0, "breaks": 1, "partly": 2, "holds": 3, "info": 4}
TAG_ORDER = {"A1": 1, "A2": 2, "A3": 3, "A4": 4, "A5": 5}


@dataclass
class Firing:
    """One rule that fired: note key, model, reasons and text fields."""

    note: str
    model: str | None = None
    reasons: list[str] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)


@cache
def load_policy() -> dict[str, Any]:
    """Load ``policy.yaml`` from the package.

    Returns
    -------
    dict
        The parsed policy.
    """
    text = files("marscope.explain").joinpath("policy.yaml").read_text(encoding="utf-8")
    data: dict[str, Any] = yaml.safe_load(text)
    return data


def policy_version() -> str:
    """Return the policy version printed in results.

    Returns
    -------
    str
        For example ``0.1.0`` or ``0.1.0-provisional``.
    """
    p = load_policy()
    return f"{p['version']}-provisional" if p.get("provisional") else str(p["version"])


def _pct(v: float) -> int:
    return round(100 * v)


def break_state(before: float, after: float, score: ModelScore) -> str:
    """Answer "does it break" from the score before and after a change.

    Parameters
    ----------
    before, after : float
        Mean clip score before and after the change.
    score : ModelScore
        The model's score (its benchmark ranges).

    Returns
    -------
    str
        ``breaks`` (falls into the benchmark real range), ``partly`` (a large drop),
        ``holds``, or ``na`` when the score was already within the real range.

    Examples
    --------
    >>> from marscope.schema import ScoreContext
    >>> class S:
    ...     context = ScoreContext(
    ...         real_rank=99,
    ...         fake_rank=50,
    ...         real_range=(0.05, 0.24),
    ...         fake_range=(0.1, 0.6),
    ...         high_fake_score=True,
    ...     )
    >>> s = S()
    >>> [break_state(0.66, after, s) for after in (0.20, 0.50, 0.62)]
    ['breaks', 'partly', 'holds']
    """
    high = score.context.real_range[1]
    if before <= high:
        return "na"
    if after <= high:
        return "breaks"
    if before - after >= float(load_policy()["cutoffs"]["partly_drop"]):
        return "partly"
    return "holds"


@cache
def benchmark() -> dict[str, Any]:
    """Return video AUROC of the released checkpoints by test set and generator.

    Returns
    -------
    dict
        ``{"sets": [...], "models": {id: {"sets", "avg", "gens"}}}``.
    """
    data: dict[str, Any] = yaml.safe_load(
        files("marscope.models").joinpath("benchmark.json").read_text(encoding="utf-8")
    )
    return data


# --- rules


def rule_a1(result: VideoResult) -> list[Firing]:
    """Ask A1 per model: does it break when the face outline is hidden."""
    firings = []
    for score in result.models.values():
        m = score.measures
        if "a1_before" not in m:
            continue
        state = break_state(m["a1_before"], m["a1_after"], score)
        fields = {"before": _pct(m["a1_before"]), "after": _pct(m["a1_after"]), "inside": _pct(m["a1_inside"])}
        firings.append(Firing(f"a1_{state}", model=score.model, fields=fields))
    return firings


def rule_a2(result: VideoResult) -> list[Firing]:
    """Ask A2 per model (does it break on a new generator) from out-of-domain AUROC."""
    sets = benchmark()["sets"]
    firings = []
    for score in result.models.values():
        bench = benchmark()["models"].get(score.model)
        if bench is None:
            continue
        values = dict(zip(sets, bench["sets"], strict=True))
        fields = {"genvideo": f"{values['GenVideo']:.1f}", "df26": f"{values['DF26']:.1f}"}
        firings.append(Firing("a2_info", model=score.model, fields=fields))
    return firings


def rule_a5(result: VideoResult) -> list[Firing]:
    """Ask A5 per model: does it break when fine pixel cues are removed."""
    firings = []
    for score in result.models.values():
        m = score.measures
        if "a5_before" not in m:
            continue
        state = break_state(m["a5_before"], m["a5_after"], score)
        fields = {"before": _pct(m["a5_before"]), "after": _pct(m["a5_after"])}
        firings.append(Firing(f"a5_{state}", model=score.model, fields=fields))
    return firings


def rule_a3(result: VideoResult) -> list[Firing]:
    """Check A3: out of scope, faces, disagreement. Shown whatever the score."""
    cut = load_policy()["cutoffs"]
    reasons = list(result.prechecks.scope_warnings)
    fields: dict[str, Any] = {}
    faces = result.prechecks.faces
    if faces is not None and faces.frames_checked > 0:
        found = faces.frames_with_face / faces.frames_checked
        if faces.frames_with_face == 0:
            reasons.append("no_face")
        elif found < cut["face_found_share"]:
            reasons.append("face_partly")
            fields["face_percent"] = round(100 * found)
        if faces.frames_with_face and faces.frames_with_several / faces.frames_with_face >= cut["several_faces_share"]:
            reasons.append("several_faces")
        if faces.frames_with_face and faces.usable_frames == 0:
            reasons.append("small_face")
    if result.agreement.disagree:
        reasons.append("disagree")
        names = {m: s.name for m, s in result.models.items()}
        fields["high"] = ", ".join(names[m] for m in result.agreement.high_fake)
        fields["other"] = ", ".join(names[m] for m in result.agreement.models if m not in result.agreement.high_fake)
    return [Firing("a3_at_risk", reasons=reasons, fields=fields)] if reasons else []


def rule_a4(result: VideoResult) -> list[Firing]:
    """Ask A4: is this copy heavily compressed; per model, does it break when shared."""
    firings = []
    bpp = result.prechecks.bits_per_pixel
    if bpp is not None and bpp < load_policy()["cutoffs"]["low_bits_per_pixel"]:
        firings.append(Firing("a4_at_risk", reasons=["low_bitrate"], fields={"bpp": f"{bpp:.3f}"}))
    for score in result.models.values():
        m = score.measures
        if "a4_before" not in m:
            continue
        state = break_state(m["a4_before"], m["a4_after"], score)
        fields = {"before": _pct(m["a4_before"]), "after": _pct(m["a4_after"])}
        firings.append(Firing(f"a4_{state}", model=score.model, fields=fields))
    return firings


def rule_provenance(result: VideoResult) -> list[Firing]:
    """Report the content credentials, and whether they contradict the scores."""
    cred = result.prechecks.credentials
    firings = []
    if cred.status == "found":
        firings.append(Firing("credentials_found", fields={"summary": cred.summary or "present"}))
        if cred.declares_ai and not result.agreement.high_fake:
            firings.append(Firing("credentials_differ"))
    elif cred.status == "absent":
        firings.append(Firing("credentials_absent"))
    elif cred.status == "unsupported":
        firings.append(Firing("credentials_unsupported"))
    return firings


def context_applies(result: VideoResult) -> bool:
    """Whether context notes show.

    When 2+ models give a high fake score, when the only model run gives one, or when
    the credentials declare AI content. "Any of three" would fire on up to about 14% of
    benchmark real videos.

    Parameters
    ----------
    result : VideoResult
        The result.

    Returns
    -------
    bool
        True if context notes apply.
    """
    high = len(result.agreement.high_fake)
    need = 1 if len(result.agreement.models) == 1 else int(load_policy()["cutoffs"]["context_min_models"])
    return high >= need or bool(result.prechecks.credentials.declares_ai)


def rule_context(result: VideoResult) -> list[Firing]:
    """Add laws and risks that may apply to deepfakes, by jurisdiction (collapsed)."""
    if not context_applies(result):
        return []
    region = result.jurisdiction
    notes = load_policy()["notes"]
    return [
        Firing(key)
        for key, note in notes.items()
        if note["kind"] == "context" and (note.get("where") == "all" or region == "all" or note.get("where") == region)
    ]


def rule_footer(result: VideoResult) -> list[Firing]:
    """Add the footer, always."""
    models = ", ".join(s.name for s in result.models.values())
    devices = ", ".join(f"{s.name} {s.device.upper()} {s.precision}" for s in result.models.values())
    return [Firing("footer", fields={"models": models, "devices": devices})]


RULES: tuple[Callable[[VideoResult], list[Firing]], ...] = (
    rule_a1,
    rule_a2,
    rule_a5,
    rule_a3,
    rule_a4,
    rule_provenance,
    rule_context,
    rule_footer,
)


# --- engine


def apply_policy(result: VideoResult, jurisdiction: str | None = None, language: str = "en") -> list[PolicyNote]:
    """Return the notes that apply to a result, merged and in a fixed order.

    Parameters
    ----------
    result : VideoResult
        A result without notes (existing notes are ignored).
    jurisdiction : {"eu", "ca", "us", "all"} or None
        Region for context notes. None shows only the notes that apply everywhere; "all"
        shows every region's notes.
    language : {"en", "fr"}
        Note language.

    Returns
    -------
    list of PolicyNote
        Trust (at risk first), provenance, context (collapsed), footer.
    """
    if jurisdiction is not None and jurisdiction not in JURISDICTIONS:
        raise ValueError(f"jurisdiction must be one of {', '.join(JURISDICTIONS)} or None")
    result = result.model_copy(update={"jurisdiction": jurisdiction})
    firings = [f for rule in RULES for f in rule(result)]
    notes = [_render(f, result, language) for f in firings]
    order = list(result.models)
    notes.sort(
        key=lambda n: (
            KIND_ORDER[n.kind],
            TAG_ORDER.get(n.tag, 9),
            order.index(n.model) if n.model in order else -1,
            STATE_ORDER[n.state],
        ),
    )
    return notes


def _render(firing: Firing, result: VideoResult, language: str) -> PolicyNote:
    policy = load_policy()
    note = policy["notes"][firing.note]
    model_name = result.models[firing.model].name if firing.model else None
    fields = {"model": model_name, **firing.fields}
    reason_texts = [_text(policy["reasons"][r], language).format(**fields) for r in firing.reasons]
    fields["reasons"] = "; ".join(reason_texts)
    return PolicyNote(
        kind=note["kind"],
        rule=firing.note,
        tag=note["tag"],
        state=note["state"],
        model=firing.model,
        title=_text(note["title"], language),
        text=_text(note["text"], language).format(**fields),
        detail=_text(note["detail"], language).format(**fields) if "detail" in note else "",
        reasons=firing.reasons,
        references=list(note.get("references", [])),
        mit=list(note.get("mit", [])),
        collapsed=note["kind"] == "context",
    )


def assess(result: VideoResult, language: str = "en") -> Reading:
    """Give the reading at the top of a result: which way the detectors lean.

    Fixed rule, from the scores' place in each model's usual ranges:

    - **fake** when every model scores above its usual range for real videos;
    - **real** when every model scores in the lower part of its real range;
    - otherwise **unclear** (the models disagree, or the scores sit in between).

    It names the model when only one ran, and never changes a score.

    Parameters
    ----------
    result : VideoResult
        The result.
    language : {"en", "fr"}
        Language of the texts.

    Returns
    -------
    Reading
        Lean, headline, one reason and the next step.
    """
    texts = load_policy()["reading"]

    def t(key: str, **values: Any) -> str:
        return _text(texts[key], language).format(**values)

    ids = list(result.models)
    names = [result.models[m].name for m in ids]
    one = len(ids) == 1
    high_ids = [m for m in ids if result.models[m].context.high_fake_score]
    join = t("and")
    if len(high_ids) == len(ids):
        lean, reason = "fake", t("one_unusual", m=names[0]) if one else t("all_unusual")
    elif not high_ids and all(result.models[m].context.real_rank < REAL_LEAN_RANK for m in ids):
        lean, reason = "real", t("one_typical", m=names[0]) if one else t("all_typical")
    elif high_ids:
        lean = "unclear"
        rest = [m for m in ids if m not in high_ids]
        groups = [
            ("fake", [result.models[m].name for m in high_ids]),
            ("real", [result.models[m].name for m in rest if result.models[m].context.real_rank < REAL_LEAN_RANK]),
            ("none", [result.models[m].name for m in rest if result.models[m].context.real_rank >= REAL_LEAN_RANK]),
        ]
        reason = " ".join(
            t(f"split_{kind}_{'one' if len(names) == 1 else 'many'}", m=_join(names, join))
            for kind, names in groups
            if names
        )
    else:
        lean, reason = "unclear", t("in_between")
    headline = t(f"lean_{lean}_one", m=names[0]) if one and lean != "unclear" else t(f"lean_{lean}")
    return Reading(
        lean=lean,  # type: ignore[arg-type]
        headline=headline,
        reasons=[reason],
        action=t("action_real" if lean == "real" else "action_check"),
    )


def _join(items: list[str], word: str) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def _text(entry: dict[str, str], language: str) -> str:
    return entry.get(language) or entry["en"]


def references(keys: list[str]) -> list[dict[str, str]]:
    """Look up references by key, in the order given, without duplicates.

    Parameters
    ----------
    keys : list of str
        Reference keys.

    Returns
    -------
    list of dict
        ``{"key", "label", "title", "url"}`` per known key.
    """
    table = load_policy()["references"]
    seen: dict[str, dict[str, str]] = {}
    for key in keys:
        if key in table and key not in seen:
            seen[key] = {"key": key, **table[key]}
    return list(seen.values())
