"""Ask: questions about one result, answered from the result and the policy notes.

The suggested questions ("chips") get fixed answers built from the result, so they work
offline and say the same thing for the same video. A free question goes to the
configured LLM with the same facts as the report and the same filter (no numbers, no
verdict, only references the rules attached). A question that asks for a verdict gets
the fixed no-verdict answer and never reaches the LLM.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from marscope.analysis.ranges import rarity
from marscope.explain.knowledge import MATCH, Card, load_cards, search
from marscope.explain.policy import assess, references
from marscope.explain.report import (
    FRENCH,
    PLAIN,
    LlmStatus,
    complete,
    endpoint_host,
    llm_payload,
    llm_settings,
    pct,
    screen,
)
from marscope.schema import Experiment, VideoResult


CHIPS = ("disagree", "tests", "certain", "verdict", "deepfake")
Chip = Literal["disagree", "tests", "certain", "verdict", "deepfake"]
# About this result: it names the video or result, or asks about a score or probability.
ABOUT_RESULT = re.compile(
    r"\b(this|the|my|our|these)\s+(video|clip|footage|result|results|score|scores|file|report|recording)\b"
    r"|\b(probability|probabilities|score|scores|percent|percentage|rated|ranked)\b"
    r"|\b(cette|la|ma|notre|ce|mon|le)\s+(vidéo|video|extrait|résultat|resultat|score|fichier|rapport)\b"
    r"|\b(probabilité|probabilite|scores?)\b",
    re.IGNORECASE,
)
Cite = Callable[..., str]  # cite("survey", ...) -> " [survey]" for the keys the rules attached
ASKS_VERDICT = re.compile(
    r"\b(is|was|are)\s+(it|this|that|the)(\s+(video|clip|person|one))?\s+(really\s+|actually\s+)?(a\s+|an\s+)?"
    r"(fake|real|deepfake|genuine|authentic|manipulated|ai|ai-generated|synthetic|legit)\b"
    r"|\b(fake|real)\s+or\s+(fake|real|not)\b|\bverdict\b"
    r"|\b(c['’]est|cette vidéo est|la vidéo est|elle est|il est)\s+(vraiment\s+)?(une?\s+)?"
    r"(faux|fausse|vrai|vraie|hypertrucage|truqu[ée]e?|authentique)\b"
    r"|\b(vrai|faux)\s+ou\s+(vrai|faux)\b|\bverdict\b",
    re.IGNORECASE,
)
ASK_PROMPT = (
    """You answer one question about one result from a video deepfake detection toolkit.
- Answer the question in the first sentence. At most three short sentences and sixty words, no headings
  or lists, no restating the question, no "the facts show".
- If the question assumes something the facts contradict (for example calls a score low when it is very
  unusual for a real video), say so first.
- "background" holds reviewed general explanations: use them for general points, and the result facts for
  anything about this video.
- If neither answers it, say so in one sentence and point to the Notes or Learn tab.
"""
    + PLAIN
)

STATE = {
    "en": {"holds": "holds", "partly": "partly holds", "breaks": "breaks", "at_risk": "is at risk"},
    "fr": {"holds": "tient", "partly": "tient en partie", "breaks": "casse", "at_risk": "est à risque"},
}
TEST = {
    "en": {
        "A1": "Face outline hidden",
        "A3": "Unlike training videos",
        "A4": "Shared many times",
        "A5": "Fine detail removed",
    },
    "fr": {
        "A1": "Contour du visage caché",
        "A3": "Loin des vidéos d’entraînement",
        "A4": "Partagée de nombreuses fois",
        "A5": "Détails fins retirés",
    },
}
TEXT = {
    "en": {
        "rank": "{m}: {label}",
        "one": "Only {m} was run, so there is nothing to compare. Pick more models on the start screen to see "
        "whether they agree.",
        "split": "They disagree: for {high}, this probability of fake is very unusual for a real video; for {low}, "
        "it is not.",
        "all_high": "They agree: every model gives a probability of fake that is very unusual for a real video.",
        "none_high": "They agree: no model gives a probability of fake that is unusual for a real video.",
        "kinds": "MARS looks at the whole frame over time, while GenD and DFD-HR look at frames one by one{c}.",
        "why_split": "When detectors disagree, the next step is to check where the video comes from{c}.",
        "why_all": "Even so, models trained on the same videos can share blind spots{c}.",
        "why_none": "A new generator can still fool all of them{c}.",
        "tests_none": "No test ran on this video: the face outline test needs a face, and the others need a "
        "high score to test.",
        "tests_line": "{t}: {s}.",
        "tests_for": " for {m}",
        "tests_how": "Breaks means the score fell to a normal level for real videos once that part was removed{c}.",
        "risk": "This video is also unlike the videos the models learned from.",
        "cred_found": "First, check where the video comes from: this file records who made it ({s}){c}.",
        "cred_none": "First, check where the video comes from: this file has no record of who made it, which is "
        "true of most videos{c}.",
        "cred_other": "First, check where the video comes from: the file's record of who made it could not be read{c}.",
        "certain_agree": "Then look for models that agree and keep their score after edits; here {state}{c}.",
        "st_split": "the models disagree",
        "st_all": "all models agree",
        "st_none": "no model finds anything unusual",
        "st_breaks": ", but the score breaks when {t}",
        "verdict": "The final decision is yours: the scores are model outputs and can be wrong{c}.",
        "verdict_scores": "Here: {r}.",
        "verdict_end": "Check where the video comes from{c}; MARScope is made for learning how detectors work{d}.",
        "and": " and ",
    },
    "fr": {
        "rank": "{m} : {label}",
        "one": "Seul {m} a été appliqué : il n’y a rien à comparer. Choisissez d’autres modèles à l’accueil pour voir "
        "s’ils sont d’accord.",
        "split": "Ils divergent : pour {high}, cette probabilité de faux est très inhabituelle pour une vraie vidéo; "
        "pour {low}, non.",
        "all_high": "Ils sont d’accord : chaque modèle donne une probabilité de faux très inhabituelle pour une "
        "vraie vidéo.",
        "none_high": "Ils sont d’accord : aucun modèle ne donne une probabilité de faux inhabituelle pour une vraie "
        "vidéo.",
        "kinds": "MARS regarde l’image entière dans le temps, GenD et DFD-HR regardent chaque image séparément{c}.",
        "why_split": "Quand les détecteurs divergent, l’étape suivante est de vérifier d’où vient la vidéo{c}.",
        "why_all": "Même ainsi, des modèles entraînés sur les mêmes vidéos peuvent partager des angles morts{c}.",
        "why_none": "Un nouveau générateur peut tout de même tous les tromper{c}.",
        "tests_none": "Aucun essai n’a tourné sur cette vidéo : l’essai du contour demande un visage, les autres un "
        "score élevé à tester.",
        "tests_line": "{t} : {s}.",
        "tests_for": " pour {m}",
        "tests_how": "Casse veut dire que le score est tombé à un niveau normal pour une vraie vidéo une fois cette "
        "partie retirée{c}.",
        "risk": "Cette vidéo ne ressemble pas non plus aux vidéos sur lesquelles les modèles ont appris.",
        "cred_found": "D’abord, vérifiez d’où vient la vidéo : ce fichier indique qui l’a créé ({s}){c}.",
        "cred_none": "D’abord, vérifiez d’où vient la vidéo : ce fichier n’indique pas qui l’a créé, comme la "
        "plupart des vidéos{c}.",
        "cred_other": "D’abord, vérifiez d’où vient la vidéo : l’indication de qui a créé le fichier est illisible{c}.",
        "certain_agree": "Cherchez ensuite des modèles d’accord qui gardent leur score quand la vidéo est modifiée; "
        "ici, {state}{c}.",
        "st_split": "les modèles divergent",
        "st_all": "tous les modèles sont d’accord",
        "st_none": "aucun modèle ne trouve rien d’inhabituel",
        "st_breaks": ", mais le score casse quand {t}",
        "verdict": "La décision finale vous revient : les scores sont des sorties de modèles et peuvent se tromper{c}.",
        "verdict_scores": "Ici : {r}.",
        "verdict_end": "Vérifiez d’où vient la vidéo{c}; MARScope sert à comprendre le fonctionnement des "
        "détecteurs{d}.",
        "and": " et ",
    },
}
# What each test removes, as a clause ("the score breaks when ...").
WHEN = {
    "en": {"A1": "the face outline is hidden", "A4": "the video is shared many times", "A5": "fine detail is removed"},
    "fr": {
        "A1": "on cache le contour du visage",
        "A4": "la vidéo est partagée de nombreuses fois",
        "A5": "on retire les détails fins",
    },
}


@dataclass
class Answer:
    """An answer to a question about a result."""

    text: str
    source: Literal["fixed", "llm", "card"]
    references: list[dict[str, str]] = field(default_factory=list)
    llm: dict[str, str] = field(default_factory=dict)
    llm_status: LlmStatus | None = None
    refused: bool = False


def asks_for_verdict(question: str) -> bool:
    """Tell whether a question asks the tool to say if the video is real or fake.

    Parameters
    ----------
    question : str
        The question.

    Returns
    -------
    bool
        True for a verdict question.

    Examples
    --------
    >>> asks_for_verdict("So is this video fake?")
    True
    >>> asks_for_verdict("Why do MARS and GenD disagree?")
    False
    >>> asks_for_verdict("Est-ce que c'est un hypertrucage ?")
    True
    """
    return bool(ASKS_VERDICT.search(question))


def _t(language: str, key: str, **values: str) -> str:
    table = TEXT.get(language, TEXT["en"])
    return table[key].format(**values)


def _join(items: list[str], word: str) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def chip_answer(result: VideoResult, chip: str, language: str = "en") -> Answer:
    """Answer a suggested question from the result alone (no LLM, same every time).

    Parameters
    ----------
    result : VideoResult
        The result, with notes.
    chip : {"disagree", "tests", "certain", "verdict"}
        The suggested question.
    language : {"en", "fr"}
        Language.

    Returns
    -------
    Answer
        Text with ``[key]`` citations, only for references the rules attached.

    Raises
    ------
    ValueError
        For an unknown chip.
    """
    lang = language if language in TEXT else "en"
    allowed = {k for n in result.notes for k in n.references}

    def cite(*keys: str) -> str:
        return "".join(f" [{k}]" for k in keys if k in allowed)

    if chip == "deepfake":
        return card_answer(next(c for c in load_cards() if c.id == "deepfake"), lang)
    builders = {"disagree": _disagree, "tests": _tests, "certain": _certain, "verdict": _verdict}
    if chip not in builders:
        raise ValueError(f"chip must be one of {', '.join(CHIPS)}")
    text = builders[chip](result, lang, cite)
    keys = [k for k in dict.fromkeys(re.findall(r"\[([a-z0-9-]+)\]", text)) if k in allowed]
    return Answer(text=text, source="fixed", references=references(keys), refused=chip == "verdict")


def card_answer(card: Card, language: str = "en") -> Answer:
    """Answer with a knowledge card's own reviewed text.

    Parameters
    ----------
    card : Card
        The card.
    language : {"en", "fr"}
        Language.

    Returns
    -------
    Answer
        The card's answer, citing the card's references.
    """
    keys = list(card.references)
    text = card.text("answer", language) + "".join(f" [{k}]" for k in keys)
    return Answer(text=text, source="card", references=references(keys))


def about_result(question: str) -> bool:
    """Tell whether a question is about this result, not about deepfakes in general.

    Parameters
    ----------
    question : str
        The question.

    Returns
    -------
    bool
        True when it names this video or result, or asks about a score or probability.

    Examples
    --------
    >>> about_result("Why is MARS's probability low?")
    True
    >>> about_result("What is a deepfake?")
    False
    """
    return bool(ABOUT_RESULT.search(question))


def _ranks(result: VideoResult, ids: list[str], lang: str) -> str:
    return "; ".join(
        _t(lang, "rank", m=result.models[m].name, label=rarity(result.models[m].context.real_rank, "", lang)[0].lower())
        for m in ids
    )


def _disagree(result: VideoResult, lang: str, cite: Cite) -> str:
    ids, a = list(result.models), result.agreement
    if len(ids) == 1:
        return _t(lang, "one", m=result.models[ids[0]].name)

    def names(ids_: list[str]) -> str:
        return _join([result.models[m].name for m in ids_], _t(lang, "and"))

    high = [m for m in ids if m in a.high_fake]
    low = [m for m in ids if m not in a.high_fake]
    if high and low:
        first = _t(lang, "split", high=names(high), low=names(low))
        why = _t(lang, "why_split", c=cite("survey"))
    elif high:
        first, why = _t(lang, "all_high"), _t(lang, "why_all", c=cite("survey"))
    else:
        first, why = _t(lang, "none_high"), _t(lang, "why_none", c=cite("survey"))
    kinds = _t(lang, "kinds", c=cite("mars")) if "mars" in ids and high and low else ""
    return " ".join(x for x in (first, kinds, why) if x)


def _test_lines(result: VideoResult, lang: str) -> list[str]:
    lines = []
    for tag in ("A1", "A4", "A5"):
        notes = [n for n in result.notes if n.kind == "trust" and n.tag == tag and n.state in STATE["en"] and n.model]
        if not notes:
            continue
        by_state: dict[str, list[str]] = {}
        for n in notes:
            by_state.setdefault(n.state, []).append(result.models[n.model].name if n.model in result.models else "")
        parts = [STATE[lang][s] + _t(lang, "tests_for", m=_join(ms, _t(lang, "and"))) for s, ms in by_state.items()]
        lines.append(_t(lang, "tests_line", t=TEST[lang][tag], s="; ".join(parts)))
    return lines


def _tests(result: VideoResult, lang: str, cite: Cite) -> str:
    lines = _test_lines(result, lang)
    out = lines or [_t(lang, "tests_none")]
    if any(n.tag == "A3" and n.state == "at_risk" for n in result.notes):
        out.append(_t(lang, "risk"))
    if any(n.kind == "trust" and n.state == "breaks" for n in result.notes):
        out.append(_t(lang, "tests_how", c=cite("survey")))
    return " ".join(out)


def _certain(result: VideoResult, lang: str, cite: Cite) -> str:
    cred = result.prechecks.credentials
    if cred.status == "found":
        first = _t(lang, "cred_found", s=cred.summary or "", c=cite("c2pa"))
    elif cred.status == "absent":
        first = _t(lang, "cred_none", c=cite("c2pa"))
    else:
        first = _t(lang, "cred_other", c=cite("c2pa"))
    a = result.agreement
    state = _t(lang, "st_split") if a.disagree else _t(lang, "st_all") if a.high_fake else _t(lang, "st_none")
    broke = [
        WHEN[lang][n.tag] for n in result.notes if n.kind == "trust" and n.state == "breaks" and n.tag in WHEN[lang]
    ]
    if broke:
        state += _t(lang, "st_breaks", t=_join(list(dict.fromkeys(broke)), _t(lang, "and")))
    return first + " " + _t(lang, "certain_agree", state=state, c=cite("survey"))


def _verdict(result: VideoResult, lang: str, cite: Cite) -> str:
    sep = " " if lang == "fr" else ""
    scores = "; ".join(
        f"{s.name} {pct(s.fake)}{sep}%, " + rarity(s.context.real_rank, "", lang)[0].lower()
        for s in result.models.values()
    )
    reading = assess(result, lang)
    return " ".join(
        [
            f"{reading.headline}: {' '.join(reading.reasons)}",
            _t(lang, "verdict", c=cite("survey")),
            _t(lang, "verdict_scores", r=scores),
            _t(lang, "verdict_end", c=cite("c2pa"), d=cite("eu-ai-act-annex3")),
        ]
    )


def ask(
    result: VideoResult,
    question: str,
    language: str = "en",
    experiments: list[Experiment] | None = None,
    timeout: float = 40.0,
) -> Answer:
    """Answer a free question with the configured LLM, from the result's facts only.

    Parameters
    ----------
    result : VideoResult
        The result, with notes.
    question : str
        The question.
    language : {"en", "fr"}
        Answer language.
    experiments : list of Experiment, optional
        Tests run in the session (sent as words, never as frames).
    timeout : float
        Seconds to wait for the LLM per attempt.

    Returns
    -------
    Answer
        The fixed no-verdict answer for a verdict question; otherwise the LLM's filtered
        answer, or an empty answer with ``llm_status`` saying why there is none.
    """
    lang = language if language in TEXT else "en"
    if asks_for_verdict(question):
        return chip_answer(result, "verdict", lang)
    hits = search(question)
    close = hits[0][0] if hits and hits[0][1] >= MATCH else None
    about = about_result(question)
    if close and not about:  # a general question: the reviewed card, never an LLM paraphrase
        return card_answer(close, lang)
    settings = llm_settings()
    if settings is None:
        strong = close is not None and hits[0][1] >= 2 * MATCH  # about this result: only a strong card match
        return card_answer(close, lang) if close and strong else Answer(text="", source="llm", llm_status="off")
    background = [c for c, score in hits if score >= MATCH / 2]
    allowed = sorted({k for n in result.notes for k in n.references} | {k for c in background for k in c.references})
    payload = llm_payload(result, experiments)
    payload["background"] = [
        {"question": c.text("question", "en"), "answer": c.text("answer", "en")} for c in background
    ]
    user = (
        "Reference keys you may cite: "
        + ", ".join(allowed)
        + "\n\nFacts:\n"
        + json.dumps(payload, indent=1)
        + "\n\nQuestion: "
        + question.strip()
    )
    raw, status = complete(ASK_PROMPT + (FRENCH if lang == "fr" else ""), user, settings, timeout=timeout)
    text = screen(raw or "", allowed)
    if not text and close is not None:  # the LLM failed or broke the rules: the closest reviewed card
        return card_answer(close, lang)
    if raw is not None and not text:
        status = "filtered"
    keys = [k for k in dict.fromkeys(re.findall(r"\[([a-z0-9-]+)\]", text)) if k in allowed]
    return Answer(
        text=text,
        source="llm",
        references=references(keys),
        llm={"model": settings["model"], "endpoint": endpoint_host(settings["base_url"])} if text else {},
        llm_status=status,
    )
