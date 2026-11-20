"""Reports: a template filled from the result, with optional LLM prose, and exports.

The template fills in every number, note and reference. The LLM, when configured, writes
only the connecting summary prose; sentences with numbers, verdicts or citations the
rules did not attach are removed, so a report cannot misstate a score. Only the
structured result, the notes and a plain description of the session's experiments go
to the LLM, never frames.

LLM settings come from the environment or a ``.env`` file::

    MARSCOPE_LLM_BASE_URL=https://.../v1
    MARSCOPE_LLM_API_KEY=...
    MARSCOPE_LLM_MODEL=Qwen3_6-35B-A3B
"""

import base64
import html
import io
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from marscope.analysis.evidence import segments
from marscope.analysis.ranges import rarity
from marscope.explain.policy import assess, references
from marscope.schema import Experiment, PolicyNote, VideoResult


log = logging.getLogger(__name__)

Language = Literal["en", "fr"]
LlmStatus = Literal["used", "off", "disabled", "timeout", "failed", "filtered"]

DISCLAIMER = {
    "en": (
        "MARScope is an educational tool for learning how video deepfake detectors work. Its scores are model "
        "outputs and can be wrong. Legal, forensic, journalistic and employment decisions, and claims about a "
        "person, need qualified experts and other evidence. Policy notes are general information, without legal "
        "advice. Provided as is under the Apache 2.0 licence, without warranty of any kind."
    ),
    "fr": (
        "MARScope est un outil pédagogique pour comprendre le fonctionnement des détecteurs d’hypertrucages vidéo. "
        "Ses scores sont des sorties de modèles et peuvent se tromper. Les décisions juridiques, judiciaires, "
        "journalistiques ou d’emploi, et toute affirmation sur une personne, demandent des experts qualifiés et "
        "d’autres éléments. Les notes de politique sont de l’information générale, sans avis juridique. Fourni tel "
        "quel sous licence Apache 2.0, sans garantie d’aucune sorte."
    ),
}
VERDICT = re.compile(
    r"\b(is|was|appears to be|seems to be|seems|looks|must be|is likely|is probably)\s+(a\s+|an\s+)?"
    r"(fake|real|deepfake|genuine|authentic|manipulated|forged|synthetic|ai-generated)\b"
    r"|\b(est|était|semble|paraît|serait|doit être)\s+(probablement\s+|sans doute\s+)?(une?\s+|être\s+)?"
    r"(faux|fausse|vrai|vraie|truqu[ée]e?|manipul[ée]e?|authentique|hypertrucage|synthétique)\b",
    re.IGNORECASE,
)
# Stricter than VERDICT, for LLM prose. A sentence is dropped if a word labels the video
# once the toolkit's own phrases ("probability of fake", "real videos") are removed, or
# if such a word sits next to an inference ("this indicates ... fake").
SAFE_PHRASES = re.compile(
    r"probability of fake|fake score|real (test )?videos?|real ones?|real or fake|fake or real|deepfake detect\w*"
    r"|probabilité de faux|vraies? vidéos?( de test)?|vrai ou faux|détecteurs? d.hypertrucages?",
    re.IGNORECASE,
)
LABEL_WORDS = re.compile(
    r"\b(fake|real|genuine|authentic|manipulated|forged|synthetic|ai-generated|faux|fausse|vrai|vraie"
    r"|truqu\w*|manipul\w*|authentique|synthétique)\b",
    re.IGNORECASE,
)
INFERENCE = re.compile(
    r"\b(indicat\w*|suggest\w*|means?|implies|imply|proves?|confirm\w*|points? to|likely|probably|may be|might be"
    r"|indique\w*|suggère\w*|signifie\w*|prouve\w*|confirme\w*|probablement)\b",
    re.IGNORECASE,
)
CITATION = re.compile(r"\[([a-z0-9-]+)\]")
# Claims that a result is evidence or can be trusted: dropped like a verdict.
EVIDENCE_CLAIM = re.compile(
    r"\b(strong|clear|conclusive|definitive|solid|reliable|firm|hard)\s+(evidence|proof|sign|indication)\b"
    r"|\b(you|we|one)\s+(can|could|may|should)\s+(fully\s+)?(trust|rely on|believe)\b"
    r"|\b(proves?|proven|confirms?|confirmed|establish(es|ed)?)\b"
    r"|\b(preuve|preuves)\s+(solide|claire|forte|concluante)s?\b|\bprouve\w*\b|\bconfirme\w*\b",
    re.IGNORECASE,
)
# A sentence about the user's tests ("No user tests were run"), dropped when none ran.
NO_TESTS = re.compile(
    r"\b(user|user's|your|additional|further|no)\b[^.]*\btests?\b(?!\s+videos?)|\btests?\b[^.]*\b(were|was)\s+(not\s+)?run\b"
    r"|\b(essais?|tests?)\b[^.]*\b(fait|faits|lancé|lancés|réalisé|réalisés)\b",
    re.IGNORECASE,
)
# Number words that can carry a score ("one" or "two" are everyday words, so not those)
NUMBER_WORDS = re.compile(
    r"\b(zero|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|percent|per\s?cent|percentage|z[ée]ro|dix|onze|douze|"
    r"treize|quatorze|quinze|seize|vingt|trente|quarante|cinquante|soixante|cent|pour\s?cent|pourcentage)\b",
    re.IGNORECASE,
)
NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)?\s?%?")
TAGS = re.compile(r"\b(?:A[1-5]|C2PA)\b")
PLAIN = """- Write for a busy executive with no technical background: everyday words, short sentences, one
  idea per sentence. Never use: AUROC, percentile, benchmark, provenance, artifact, residual, calibrated,
  inference, or a test code (A1 to A5) without its plain name.
- Call a score the "probability of fake". Whether it is high or low is only what
  "compared_with_real_videos" says for that model (for example "very unusual for a real video").
- Use no numbers at all, in digits or in words; the app shows every number.
- Never say or imply that the video is real, fake, a deepfake, authentic or manipulated, and never say
  where a forgery is: describe what the models found and what they use.
- You may cite only the reference keys given, in square brackets, like [survey]. Use only the facts given."""
SYSTEM_PROMPT = (
    """You write the summary at the top of a report from a video deepfake detection toolkit.
- At most three short sentences and sixty words, no headings or lists: first what the models found,
  then what the user's tests showed, then what limits the result (scores can be wrong, models can
  disagree, sharing can change a score). If "tests_the_user_ran" is empty,
  do not mention tests at all.
- A good first sentence: "All three models give a probability of fake that is very unusual for a real
  video."
"""
    + PLAIN
)
MAX_SENTENCES, MAX_WORDS = 3, 70  # enforced on every LLM answer, whatever the prompt says
FRENCH = "\n- Write in Canadian French."

TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "MARScope report: {name}",
        "tagline": "An educational tool. Scores are model outputs and can be wrong.",
        "summary": "Summary",
        "scores": "Scores in context",
        "th_model": "Model",
        "th_fake": "Probability of fake",
        "th_rank": "Compared with real videos",
        "th_ranges": "Real videos usually",
        "th_device": "Run on",
        "rank": "{label}: {odds}",
        "ranges": "{a}–{b}%",
        "tried": "What you tried",
        "tried_intro": "Tests run in the app during this session, with the scores the toolkit measured.",
        "notes": "Notes",
        "h_trust": "How far to trust the models",
        "h_prov": "Where the video comes from",
        "h_ctx": "Laws and risks that may apply",
        "h_foot": "About this result",
        "refs": "References",
        "versions": "Versions",
        "disclaimer": "Disclaimer",
        "by_used": "Summary written by {model} ({endpoint}) from the result. Every number, note and reference comes "
        "from the template.",
        "by_off": "Summary from the template: no language model is set up.",
        "by_disabled": "Summary from the template.",
        "by_timeout": "The language model did not answer in time, so the summary is the template version.",
        "by_failed": "The language model could not be reached, so the summary is the template version.",
        "by_filtered": "The language model's summary broke the report rules, so the template version is shown.",
        "s_run_one": "{names} was run on this video.",
        "s_run_many": "{names} were run on this video.",
        "s_none": "No model gave a probability of fake that is unusual for a real video.",
        "s_all_one": "{high} gave a probability of fake that is very unusual for a real video.",
        "s_all_many": "{high} all gave a probability of fake that is very unusual for a real video.",
        "s_some": "{high} gave a probability of fake that is very unusual for a real video; the others did not.",
        "s_tried_one": "One test you ran in the app is listed under What you tried.",
        "s_tried_many": "The {n} tests you ran in the app are listed under What you tried.",
        "s_end": "The notes below say how far to trust each model.",
        "and": " and ",
        "x_change": "{m} {a}% to {b}%",
        "x_frames": "Clip {c}: with {n} frames hidden (frames {frames}), the scores moved: {changes}.",
        "x_frames_one": "Clip {c}: with frame {frames} hidden, the scores moved: {changes}.",
        "x_frames_test": "Clip {c}: hiding {m}’s top {n} frames moved its score from {a}% to {b}%; hiding {n} frames "
        "at random moved it to {r}%.",
        "x_areas_test": "Clip {c}: hiding {m}’s top areas in every frame moved its score from {a}% to {b}%; hiding "
        "as many areas at random moved it to {r}%.",
        "x_occ_flat": "Clip {c}: an occlusion map for {m} found no area whose hiding changes its score.",
        "x_occ": "Clip {c}: an occlusion map for {m} showed which areas change its score when hidden (shown in the "
        "app).",
        "x_edit": "Clip {c}, edited ({settings}): the scores moved: {changes}.",
        "jpeg": "JPEG quality {v}",
        "scale": "{v}% resolution",
        "blur": "blur {v} px",
        "noise": "noise {v}",
        "frame_cap": "Frame {n} (clip {c}), as the models saw it, before any edit.",
    },
    "fr": {
        "title": "Rapport MARScope : {name}",
        "tagline": "Un outil pédagogique. Les scores sont des sorties de modèles et peuvent se tromper.",
        "summary": "Résumé",
        "scores": "Les scores en contexte",
        "th_model": "Modèle",
        "th_fake": "Probabilité de faux",
        "th_rank": "Par rapport aux vraies vidéos",
        "th_ranges": "Vraies vidéos, d’habitude",
        "th_device": "Exécuté sur",
        "rank": "{label} : {odds}",
        "ranges": "{a}–{b} %",
        "tried": "Ce que vous avez essayé",
        "tried_intro": "Essais faits dans l’application pendant cette session, avec les scores mesurés par l’outil.",
        "notes": "Notes",
        "h_trust": "Jusqu’où se fier aux modèles",
        "h_prov": "D’où vient la vidéo",
        "h_ctx": "Lois et risques qui peuvent s’appliquer",
        "h_foot": "À propos de ce résultat",
        "refs": "Références",
        "versions": "Versions",
        "disclaimer": "Avertissement",
        "by_used": "Résumé rédigé par {model} ({endpoint}) à partir du résultat. Chaque nombre, note et référence "
        "vient du modèle de rapport.",
        "by_off": "Résumé du modèle de rapport : aucun modèle de langage n’est configuré.",
        "by_disabled": "Résumé du modèle de rapport.",
        "by_timeout": "Le modèle de langage n’a pas répondu à temps : voici la version du modèle de rapport.",
        "by_failed": "Le modèle de langage est injoignable : voici la version du modèle de rapport.",
        "by_filtered": "Le résumé du modèle de langage ne respectait pas les règles du rapport : voici la version du "
        "modèle de rapport.",
        "s_run_one": "{names} a été appliqué à cette vidéo.",
        "s_run_many": "{names} ont été appliqués à cette vidéo.",
        "s_none": "Aucun modèle n’a donné une probabilité de faux inhabituelle pour une vraie vidéo.",
        "s_all_one": "{high} a donné une probabilité de faux très inhabituelle pour une vraie vidéo.",
        "s_all_many": "{high} ont tous donné une probabilité de faux très inhabituelle pour une vraie vidéo.",
        "s_some": "{high} a donné une probabilité de faux très inhabituelle pour une vraie vidéo; les autres non.",
        "s_tried_one": "L’essai que vous avez fait dans l’application figure sous Ce que vous avez essayé.",
        "s_tried_many": "Les {n} essais faits dans l’application figurent sous Ce que vous avez essayé.",
        "s_end": "Les notes ci-dessous disent jusqu’où se fier à chaque modèle.",
        "and": " et ",
        "x_change": "{m} {a} % à {b} %",
        "x_frames": "Extrait {c} : avec {n} images cachées (images {frames}), les scores ont changé : {changes}.",
        "x_frames_one": "Extrait {c} : avec l’image {frames} cachée, les scores ont changé : {changes}.",
        "x_frames_test": "Extrait {c} : cacher les {n} meilleures images de {m} fait passer son score de {a} % à "
        "{b} %; cacher {n} images au hasard le fait passer à {r} %.",
        "x_areas_test": "Extrait {c} : cacher les zones principales de {m} dans chaque image fait passer son score de "
        "{a} % à {b} %; cacher autant de zones au hasard le fait passer à {r} %.",
        "x_occ_flat": "Extrait {c} : une carte d’occlusion de {m} ne trouve aucune zone dont le masquage change son "
        "score.",
        "x_occ": "Extrait {c} : une carte d’occlusion de {m} a montré quelles zones changent son score quand on les "
        "cache (affichée dans l’application).",
        "x_edit": "Extrait {c}, modifié ({settings}) : les scores ont changé : {changes}.",
        "jpeg": "qualité JPEG {v}",
        "scale": "résolution {v} %",
        "blur": "flou {v} px",
        "noise": "bruit {v}",
        "frame_cap": "Image {n} (extrait {c}), telle que les modèles l’ont vue, avant toute modification.",
    },
}


def tr(language: str, key: str, **values: Any) -> str:
    """Return a report string in the language, filled in.

    Parameters
    ----------
    language : {"en", "fr"}
        Language (anything else falls back to English).
    key : str
        String key.
    **values
        Placeholders.

    Returns
    -------
    str
        The text.

    Examples
    --------
    >>> tr("fr", "ranges", a=7, b=24)
    '7–24 %'
    """
    table = TEXT.get(language, TEXT["en"])
    return table.get(key, TEXT["en"][key]).format(**values)


def pct(value: float) -> int:
    """Return a score as a whole percentage, as the UI shows it.

    Parameters
    ----------
    value : float
        A score between 0 and 1.

    Returns
    -------
    int
        The percentage, rounded half up.

    Examples
    --------
    >>> pct(0.665), pct(1.0)
    (67, 100)
    """
    return int(value * 100 + 0.5)


@dataclass
class Report:
    """A report, ready to export."""

    title: str
    summary: str
    summary_source: Literal["llm", "template"]
    rows: list[dict[str, str]]
    notes: list[PolicyNote]
    refs: list[dict[str, str]]
    versions: dict[str, str]
    experiments: list[str] = field(default_factory=list)
    language: str = "en"
    llm: dict[str, str] = field(default_factory=dict)
    llm_status: LlmStatus = "disabled"
    payload: dict[str, Any] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER["en"]
    frame: bytes | None = None  # JPEG of one kept frame, only when the user asks for it (HTML and PDF)
    frame_caption: str = ""

    @property
    def by_line(self) -> str:
        """Who wrote the summary, in the report's language."""
        return tr(self.language, "by_" + self.llm_status, **self.llm)

    def text(self, key: str) -> str:
        """Return a heading or label in the report's language.

        Parameters
        ----------
        key : str
            String key.

        Returns
        -------
        str
            The text.
        """
        return tr(self.language, key)


# --- build


def build_report(
    result: VideoResult,
    use_llm: bool = True,
    experiments: list[Experiment] | None = None,
    language: str = "en",
    timeout: float = 40.0,
) -> Report:
    """Build a report from a result.

    Parameters
    ----------
    result : VideoResult
        An analyzed result, with notes.
    use_llm : bool
        Ask the configured LLM for the summary prose. Falls back to the template if it
        is not configured, fails, or breaks the rules.
    experiments : list of Experiment, optional
        Tests run in the session, listed under "What you tried".
    language : {"en", "fr"}
        Report language (the notes keep the language they were written in).
    timeout : float
        Seconds to wait for the LLM per attempt.

    Returns
    -------
    Report
        The report.
    """
    experiments = experiments or []
    lang = language if language in TEXT else "en"
    allowed = sorted({key for note in result.notes for key in note.references})
    summary, source, llm_info = template_summary(result, lang, len(experiments)), "template", {}
    payload = llm_payload(result, experiments)
    status: LlmStatus = "disabled"
    if use_llm:
        settings = llm_settings()
        status = "off"
        if settings is not None:
            system = SYSTEM_PROMPT + (FRENCH if lang == "fr" else "")
            user = "Reference keys you may cite: " + ", ".join(allowed) + "\n\nFacts:\n" + json.dumps(payload, indent=1)
            text, status = complete(system, user, settings, timeout=timeout)
            raw = text or ""
            if not experiments:  # models often mention "no tests" despite the prompt; such a sentence adds nothing
                raw = " ".join(x for x in re.split(r"(?<=[.!?])\s+", raw) if not NO_TESTS.search(x))
            prose = screen(raw, allowed, min_sentences=2)
            if prose:
                summary, source = prose, "llm"
                llm_info = {"model": settings["model"], "endpoint": endpoint_host(settings["base_url"])}
            elif status == "used":
                status = "filtered"
    rows = []
    for s in result.models.values():
        ra, rb = s.context.real_range
        rows.append(
            {
                "model": s.name,
                "fake": f"{pct(s.fake)}%",
                "rank": tr(lang, "rank", **dict(zip(("label", "odds"), rarity(s.context.real_rank, s.name, lang)))),
                "ranges": tr(lang, "ranges", a=pct(ra), b=pct(rb)),
                "device": f"{s.device.upper()} {s.precision}",
            }
        )
    versions = {
        "package": result.versions.package,
        "protocol": result.versions.protocol,
        "policy": result.versions.policy,
        **{f"checkpoint {m}": c for m, c in result.versions.checkpoints.items()},
    }
    return Report(
        title=tr(lang, "title", name=result.video.name),
        summary=summary,
        summary_source=source,  # type: ignore[arg-type]
        rows=rows,
        notes=result.notes,
        refs=references(allowed),
        versions=versions,
        experiments=[describe_experiment(e, result, lang) for e in experiments],
        language=lang,
        llm=llm_info,
        llm_status=status,
        payload=payload,
        disclaimer=DISCLAIMER[lang],
    )


def template_summary(result: VideoResult, language: str = "en", n_experiments: int = 0) -> str:
    """Write the fixed summary used without an LLM.

    Parameters
    ----------
    result : VideoResult
        The result.
    language : {"en", "fr"}
        Language.
    n_experiments : int
        Tests run in the session.

    Returns
    -------
    str
        Three or four sentences.
    """
    names = [s.name for s in result.models.values()]
    high = [result.models[m].name for m in result.agreement.high_fake]
    join = tr(language, "and")
    reading = assess(result, language) if result.notes else None
    parts = [f"{reading.headline}." if reading else ""]
    parts.append(tr(language, "s_run_one" if len(names) == 1 else "s_run_many", names=_join(names, join)))
    if not high:
        parts.append(tr(language, "s_none"))
    elif len(high) == len(names):
        parts.append(tr(language, "s_all_one" if len(high) == 1 else "s_all_many", high=_join(high, join)))
    else:
        parts.append(tr(language, "s_some", high=_join(high, join)))
    if n_experiments:
        parts.append(tr(language, "s_tried_one" if n_experiments == 1 else "s_tried_many", n=n_experiments))
    parts.append(tr(language, "s_end"))
    return " ".join(x for x in parts if x)


def _join(items: list[str], word: str = " and ") -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + word + items[-1]


def shown_frames(result: VideoResult, clip: int, runs: list[int]) -> list[int]:
    """Return the frame numbers the UI shows for runs of a clip (1-based, as on screen).

    Parameters
    ----------
    result : VideoResult
        The result.
    clip : int
        Clip number.
    runs : list of int
        Runs of the clip (16 per clip, fewer on short clips).

    Returns
    -------
    list of int
        The middle frame of each run, numbered from 1 in the original video.
    """
    ref = next(iter(result.models.values())).clips[clip]
    segs = segments(ref.n_valid)
    stride = result.video.frame_stride
    return [ref.first_frame + segs[r][(len(segs[r]) - 1) // 2] * stride + 1 for r in sorted(runs) if r < len(segs)]


def describe_experiment(exp: Experiment, result: VideoResult, language: str = "en") -> str:
    """Describe one session test in a plain sentence, with its measured scores.

    Parameters
    ----------
    exp : Experiment
        The test.
    result : VideoResult
        The result it was run on (for model names).
    language : {"en", "fr"}
        Language.

    Returns
    -------
    str
        One sentence.
    """
    name = result.models[exp.model].name if exp.model and exp.model in result.models else (exp.model or "")
    c = exp.clip + 1
    changes = ", ".join(
        tr(language, "x_change", m=result.models[m].name, a=pct(exp.before[m]), b=pct(exp.after[m]))
        for m in exp.after
        if m in result.models and m in exp.before
    )
    if exp.kind == "frames":
        key = "x_frames_one" if exp.hidden == 1 else "x_frames"
        return tr(language, key, c=c, n=exp.hidden, frames=", ".join(map(str, exp.frames)), changes=changes)
    if exp.kind in ("frames_test", "areas_test") and exp.model:
        a, b = pct(exp.before.get(exp.model, 0.0)), pct(exp.after.get(exp.model, 0.0))
        r = pct(exp.random_after or 0.0)
        return tr(language, "x_" + exp.kind, c=c, m=name, n=exp.hidden, a=a, b=b, r=r)
    if exp.kind == "occlusion":
        return tr(language, "x_occ_flat" if exp.flat else "x_occ", c=c, m=name)
    defaults = {"jpeg": 100.0, "scale": 100.0, "blur": 0.0, "noise": 0.0}
    settings = ", ".join(
        tr(language, k, v=f"{v:g}") for k, v in exp.settings.items() if k in defaults and v != defaults[k]
    )
    return tr(language, "x_edit", c=c, settings=settings, changes=changes)


# --- LLM


def llm_settings() -> dict[str, str] | None:
    """Read the LLM settings from the environment (after loading ``.env``).

    The ``.env`` file is looked for in the folder MARScope runs from (and its parents),
    or at ``$MARSCOPE_ENV_FILE`` (set by ``--env-file``). Variables already set in the
    environment win over the file.

    Returns
    -------
    dict or None
        ``base_url``, ``api_key`` and ``model``, or None if any is missing.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # noqa: PLC0415

        # usecwd: an installed package looks where the user runs it.
        load_dotenv(os.environ.get("MARSCOPE_ENV_FILE") or find_dotenv(usecwd=True))
    except ImportError:  # pragma: no cover - python-dotenv is a dependency
        pass
    settings = {
        "base_url": os.environ.get("MARSCOPE_LLM_BASE_URL", ""),
        "api_key": os.environ.get("MARSCOPE_LLM_API_KEY", ""),
        "model": os.environ.get("MARSCOPE_LLM_MODEL") or "Qwen3_6-35B-A3B",  # empty in .env means the default
    }
    return settings if settings["base_url"] and settings["api_key"] else None


def endpoint_host(base_url: str) -> str:
    """Return the host of an endpoint URL, for the "written by" line.

    Parameters
    ----------
    base_url : str
        The endpoint.

    Returns
    -------
    str
        Host name only.

    Examples
    --------
    >>> endpoint_host("https://llm.example.org/v1")
    'llm.example.org'
    """
    return re.sub(r"^https?://", "", base_url).split("/")[0]


def _effect(before: float, after: float) -> str:
    change = after - before
    size = abs(change)
    if size < 0.05:
        return "little change"
    return ("large " if size >= 0.2 else "moderate ") + ("drop" if change < 0 else "rise")


def llm_payload(result: VideoResult, experiments: list[Experiment] | None = None) -> dict[str, Any]:
    """Return the facts sent to the LLM: scores, notes and tests in words, never frames.

    Parameters
    ----------
    result : VideoResult
        The result.
    experiments : list of Experiment, optional
        Tests run in the session.

    Returns
    -------
    dict
        JSON-serializable facts, with no raw score.
    """
    names = {m: s.name for m, s in result.models.items()}
    tests = []
    for e in experiments or []:
        entry: dict[str, Any] = {"test": e.kind.replace("_", " "), "clip": e.clip + 1}
        if e.model:
            entry["model"] = names.get(e.model, e.model)
        if e.kind in ("frames_test", "areas_test") and e.model in e.after:
            before, after = e.before.get(e.model, 0.0), e.after[e.model]
            entry["effect"] = _effect(before, after)
            entry["random_effect"] = _effect(before, e.random_after or before)
        elif e.kind == "occlusion":
            entry["effect"] = "no area changes the score" if e.flat else "some areas change the score"
        else:
            entry["effect"] = {
                names[m]: _effect(e.before[m], v) for m, v in e.after.items() if m in e.before and m in names
            }
        tests.append(entry)
    return {
        "models": [
            {
                "name": s.name,
                "fake_score_higher_than_it_usually_gives_real_videos": s.context.high_fake_score,
                "compared_with_real_videos": rarity(s.context.real_rank)[0].lower(),
                "device": s.device,
            }
            for s in result.models.values()
        ],
        "models_disagree": result.agreement.disagree,
        "evidence_strength": result.agreement.strength,
        "notes": [
            {"note": NUMBER.sub("#", n.text), "references": n.references} for n in result.notes if n.kind != "footer"
        ],
        "tests_the_user_ran": tests,
    }


def complete(
    system: str,
    user: str,
    settings: dict[str, str],
    *,
    timeout: float = 40.0,
    attempts: int = 2,
    max_tokens: int = 500,
) -> tuple[str | None, LlmStatus]:
    """Ask the configured OpenAI-compatible endpoint once, with a retry.

    Parameters
    ----------
    system : str
        System prompt.
    user : str
        User message (facts only, never frames).
    settings : dict
        From :func:`llm_settings`.
    timeout : float
        Seconds per attempt.
    attempts : int
        Tries before giving up (a rejected key is not retried).
    max_tokens : int
        Answer length limit.

    Returns
    -------
    tuple of (str or None, str)
        The raw answer and ``"used"``, or None and ``"timeout"`` / ``"failed"``.
    """
    from openai import APITimeoutError, AuthenticationError, OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=settings["api_key"], base_url=settings["base_url"], timeout=timeout, max_retries=0)
    status: LlmStatus = "failed"
    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=settings["model"],
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.2,
                max_tokens=max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            return response.choices[0].message.content or "", "used"
        except AuthenticationError as error:
            log.warning("LLM rejected the key: %s", error)
            return None, "failed"
        except APITimeoutError:
            status = "timeout"
            log.warning("LLM timed out (%s/%s)", attempt + 1, attempts)
        except Exception as error:  # noqa: BLE001 - any failure falls back to the template
            status = "failed"
            log.warning("LLM call failed (%s/%s): %s", attempt + 1, attempts, error)
        if attempt < attempts - 1:
            time.sleep(1)
    return None, status


def labels_video(sentence: str) -> bool:
    """Tell whether a sentence says or implies what the video is (real, fake, ...).

    Parameters
    ----------
    sentence : str
        One sentence of LLM prose.

    Returns
    -------
    bool
        True if the sentence must be dropped.

    Examples
    --------
    >>> labels_video("This indicates a high probability of fake content.")
    True
    >>> labels_video("This suggests the content may be synthetic.")
    True
    >>> labels_video("You can trust these findings as strong evidence.")
    True
    >>> labels_video("Its probability of fake is very unusual for a real video.")
    False
    """
    if (
        VERDICT.search(sentence)
        or EVIDENCE_CLAIM.search(sentence)
        or LABEL_WORDS.search(SAFE_PHRASES.sub(" ", sentence))
    ):
        return True
    return bool(INFERENCE.search(sentence) and LABEL_WORDS.search(sentence))


def screen(text: str, allowed: list[str], min_sentences: int = 1) -> str:
    """Filter, then shorten, an LLM answer, or reject it if the filter gutted it.

    The answer is rejected (empty string) when its first sentence, the bottom line, is
    dropped, or when fewer than ``min_sentences`` sentences survive.

    Parameters
    ----------
    text : str
        Raw LLM output.
    allowed : list of str
        Reference keys that may be cited.
    min_sentences : int
        Fewest sentences an answer must keep.

    Returns
    -------
    str
        The safe, capped answer, or an empty string.

    Examples
    --------
    >>> screen("The video is fake. One score is not proof.", [])
    ''
    >>> screen("One score is not proof. Check the source.", [], min_sentences=2)
    'One score is not proof. Check the source.'
    """
    raw = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    sentences = [x for x in re.split(r"(?<=[.!?])\s+", raw) if x.strip()]
    dropped = [x for x in sentences if not clean_prose(x, allowed)]
    if dropped:
        log.debug("LLM sentences dropped by the filter: %s", dropped)
    if not sentences or not clean_prose(sentences[0], allowed):
        return ""
    kept = cap(clean_prose(raw, allowed))
    if len([x for x in re.split(r"(?<=[.!?])\s+", kept) if x.strip()]) < min_sentences:
        return ""
    return kept


def cap(text: str, sentences: int = MAX_SENTENCES, words: int = MAX_WORDS) -> str:
    """Keep the first sentences of an answer, within a sentence and word budget.

    Parameters
    ----------
    text : str
        Cleaned prose.
    sentences : int
        Most sentences kept.
    words : int
        Most words kept (whole sentences only; the first one is always kept).

    Returns
    -------
    str
        The shortened prose.

    Examples
    --------
    >>> cap("One. Two. Three. Four.", sentences=3)
    'One. Two. Three.'
    """
    kept: list[str] = []
    for sentence in re.split(r"(?<=[.!?])\s+", text.strip()):
        if not sentence:
            continue
        if kept and (len(kept) >= sentences or len(" ".join([*kept, sentence]).split()) > words):
            break
        kept.append(sentence)
    return " ".join(kept)


def clean_prose(text: str, allowed: list[str]) -> str:
    """Keep only safe sentences from LLM prose.

    Removes reasoning blocks, citations outside ``allowed``, and sentences with numbers
    (digits or words) or a verdict (test names A1 to A5 and C2PA are not numbers).

    Parameters
    ----------
    text : str
        LLM output.
    allowed : list of str
        Reference keys the rules attached.

    Returns
    -------
    str
        Cleaned prose (may be empty).

    Examples
    --------
    >>> text = "MARS uses edges [mars]. The video is fake. It got 0.9. It is forty."
    >>> clean_prose(text, ["mars"])
    'MARS uses edges [mars].'
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = CITATION.sub(lambda m: m.group(0) if m.group(1) in allowed else "", text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    keep = []
    for sentence in sentences:
        bare = TAGS.sub("", CITATION.sub("", sentence))
        if not sentence.strip() or re.search(r"\d", bare) or NUMBER_WORDS.search(bare) or labels_video(bare):
            continue
        keep.append(re.sub(r"\s+([.,;])", r"\1", sentence.strip()))
    return " ".join(keep)


# --- export

NOTE_GROUPS = (("trust", "h_trust"), ("provenance", "h_prov"), ("context", "h_ctx"), ("footer", "h_foot"))
COLUMNS = (
    ("model", "th_model"),
    ("fake", "th_fake"),
    ("rank", "th_rank"),
    ("ranges", "th_ranges"),
    ("device", "th_device"),
)


def to_markdown(report: Report) -> str:
    """Render a report as Markdown.

    Parameters
    ----------
    report : Report
        The report.

    Returns
    -------
    str
        Markdown text.
    """
    t = report.text
    lines = [f"# {report.title}", "", f"*{t('tagline')}*", "", f"## {t('summary')}", "", report.summary, ""]
    lines += [f"*{report.by_line}*", ""]
    lines += [f"## {t('scores')}", "", "| " + " | ".join(t(k) for _, k in COLUMNS) + " |", "|---" * len(COLUMNS) + "|"]
    lines += ["| " + " | ".join(r[c] for c, _ in COLUMNS) + " |" for r in report.rows]
    if report.experiments:
        lines += ["", f"## {t('tried')}", "", t("tried_intro"), ""] + [f"- {x}" for x in report.experiments]
    lines += ["", f"## {t('notes')}", ""]
    for kind, heading in NOTE_GROUPS:
        notes = [n for n in report.notes if n.kind == kind]
        if notes:
            lines += [f"### {t(heading)}", ""]
            lines += [f"- **{n.text}**{_cite(n)}" + (f"  \n  {n.detail}" if n.detail else "") for n in notes]
            lines.append("")
    if report.refs:
        lines += [f"## {t('refs')}", ""]
        lines += [f"- [{r['key']}] {r['label']}: {r['title']}. {r['url']}" for r in report.refs]
        lines.append("")
    lines += [f"## {t('versions')}", ""] + [f"- {k}: {v}" for k, v in report.versions.items()]
    lines += ["", f"## {t('disclaimer')}", "", report.disclaimer, ""]
    return "\n".join(lines)


def _cite(note: PolicyNote) -> str:
    return " " + " ".join(f"[{k}]" for k in note.references) if note.references else ""


def to_html(report: Report) -> str:
    """Render a report as a standalone HTML page with a print stylesheet.

    Parameters
    ----------
    report : Report
        The report.

    Returns
    -------
    str
        HTML text.
    """
    e, t = html.escape, report.text
    head = "".join(f"<th>{e(t(k))}</th>" for _, k in COLUMNS)
    rows = "".join("<tr>" + "".join(f"<td>{e(r[c])}</td>" for c, _ in COLUMNS) + "</tr>" for r in report.rows)
    tried = ""
    if report.experiments:
        tried = (
            f'<h2>{e(t("tried"))}</h2><p class="note">{e(t("tried_intro"))}</p><ul>'
            + "".join(f"<li>{e(x)}</li>" for x in report.experiments)
            + "</ul>"
        )
    notes = ""
    for kind, heading in NOTE_GROUPS:
        items = [n for n in report.notes if n.kind == kind]
        if items:
            notes += (
                f"<h3>{e(t(heading))}</h3><ul>"
                + "".join(
                    f"<li><strong>{e(n.text)}</strong>{e(_cite(n))}"
                    + (f'<br><span class="note">{e(n.detail)}</span>' if n.detail else "")
                    + "</li>"
                    for n in items
                )
                + "</ul>"
            )
    refs = "".join(
        f'<li>[{e(r["key"])}] {e(r["label"])}: <a href="{e(r["url"])}">{e(r["title"])}</a></li>' for r in report.refs
    )
    versions = "".join(f"<li>{e(k)}: {e(v)}</li>" for k, v in report.versions.items())
    figure = ""
    if report.frame:
        data = base64.b64encode(report.frame).decode("ascii")
        figure = (
            f'<figure><img src="data:image/jpeg;base64,{data}" alt="{e(report.frame_caption)}">'
            f'<figcaption class="note">{e(report.frame_caption)}</figcaption></figure>'
        )
    return f"""<!doctype html>
<html lang="{e(report.language)}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(report.title)}</title>
<style>
body{{font:15px/1.55 "Open Sans",-apple-system,"Helvetica Neue",Arial,sans-serif;color:#15140f;background:#fff;max-width:860px;margin:32px auto;padding:0 16px}}
h1{{font-size:26px}} h2{{font-size:19px;margin-top:28px}} h3{{font-size:15px;margin-bottom:4px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{border-bottom:1px solid #e4e4e0;padding:6px;text-align:left;vertical-align:top}}
.note{{color:#6e6e73}} figure{{margin:16px 0}} img{{max-width:100%;border-radius:12px}} @media print{{body{{margin:0}} a{{color:inherit}}}}
</style></head><body>
<h1>{e(report.title)}</h1><p class="note"><em>{e(t("tagline"))}</em></p>
<h2>{e(t("summary"))}</h2><p>{e(report.summary)}</p><p class="note"><em>{e(report.by_line)}</em></p>
<h2>{e(t("scores"))}</h2><table><tr>{head}</tr>{rows}</table>
{figure}
{tried}
<h2>{e(t("notes"))}</h2>{notes}
<h2>{e(t("refs"))}</h2><ul>{refs}</ul>
<h2>{e(t("versions"))}</h2><ul>{versions}</ul>
<h2>{e(t("disclaimer"))}</h2><p class="note">{e(report.disclaimer)}</p>
</body></html>
"""


def to_pdf(report: Report) -> bytes:
    """Render a report as PDF with ReportLab.

    Parameters
    ----------
    report : Report
        The report.

    Returns
    -------
    bytes
        PDF file content.
    """
    from reportlab.lib import colors  # noqa: PLC0415
    from reportlab.lib.pagesizes import A4  # noqa: PLC0415
    from reportlab.lib.styles import getSampleStyleSheet  # noqa: PLC0415
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: PLC0415

    styles = getSampleStyleSheet()
    e, t = html.escape, report.text
    small = styles["BodyText"].clone("small", fontSize=8, leading=10)
    story: list[Any] = [
        Paragraph(e(report.title), styles["Title"]),
        Paragraph(f"<i>{e(t('tagline'))}</i>", styles["Normal"]),
    ]
    story += [
        Spacer(1, 8),
        Paragraph(e(t("summary")), styles["Heading2"]),
        Paragraph(e(report.summary), styles["Normal"]),
    ]
    story += [Spacer(1, 4), Paragraph(f"<i>{e(report.by_line)}</i>", small)]
    data = [[Paragraph(e(t(k)), small) for _, k in COLUMNS]] + [
        [Paragraph(e(r[c]), small) for c, _ in COLUMNS] for r in report.rows
    ]
    table = Table(data, colWidths=[55, 50, 50, 130, 140, 55])
    table.setStyle(
        TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")])
    )
    story += [Paragraph(e(t("scores")), styles["Heading2"]), table]
    if report.frame:
        from PIL import Image as PILImage  # noqa: PLC0415
        from reportlab.platypus import Image  # noqa: PLC0415

        with PILImage.open(io.BytesIO(report.frame)) as img:
            ratio = img.height / img.width
        story += [Spacer(1, 8), Image(io.BytesIO(report.frame), width=440, height=440 * ratio)]
        story.append(Paragraph(e(report.frame_caption), small))
    if report.experiments:
        story += [Paragraph(e(t("tried")), styles["Heading2"]), Paragraph(e(t("tried_intro")), small)]
        story += [Paragraph("• " + e(x), styles["Normal"]) for x in report.experiments]
    story.append(Paragraph(e(t("notes")), styles["Heading2"]))
    for kind, heading in NOTE_GROUPS:
        items = [n for n in report.notes if n.kind == kind]
        if items:
            story.append(Paragraph(e(t(heading)), styles["Heading3"]))
        for n in items:
            story.append(Paragraph(f"<b>{e(n.text)}</b>{e(_cite(n))}", styles["Normal"]))
            if n.detail:
                story.append(Paragraph(e(n.detail), small))
            story.append(Spacer(1, 4))
    if report.refs:
        story.append(Paragraph(e(t("refs")), styles["Heading2"]))
        story += [Paragraph(e(f"[{r['key']}] {r['label']}: {r['title']}. {r['url']}"), small) for r in report.refs]
    story.append(Paragraph(e(t("versions")), styles["Heading2"]))
    story += [Paragraph(e(f"{k}: {v}"), small) for k, v in report.versions.items()]
    story += [Paragraph(e(t("disclaimer")), styles["Heading2"]), Paragraph(e(report.disclaimer), small)]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, title=report.title).build(story)
    return buffer.getvalue()
