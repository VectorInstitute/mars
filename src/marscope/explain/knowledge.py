"""Knowledge cards: short, reviewed answers to general questions about deepfakes.

The cards live in ``knowledge.yaml`` (English and French, with references). Ask finds
the closest cards with a small BM25 search over each card's question, keywords and
answer: pure Python, no model, so it works offline and gives the same match every time.
"""

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import cache
from importlib.resources import files
from typing import Any

import yaml


# A best score at or above this is a close match (set from the questions in the tests).
MATCH = 3.0
FIELD_WEIGHTS = {"q": 3, "keywords": 3, "answer": 1}
STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "when",
        "where",
        "can",
        "could",
        "should",
        "would",
        "will",
        "i",
        "you",
        "we",
        "our",
        "my",
        "your",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "from",
        "by",
        "about",
        "as",
        "at",
        "or",
        "and",
        "not",
        "no",
        "me",
        "us",
        "they",
        "them",
        "there",
        "here",
        "than",
        "then",
        "so",
        "if",
        "into",
        "any",
        "some",
        "tell",
        "explain",
        "please",
        "un",
        "une",
        "le",
        "la",
        "les",
        "l",
        "de",
        "des",
        "du",
        "d",
        "est",
        "sont",
        "que",
        "qu",
        "qui",
        "quoi",
        "comment",
        "pourquoi",
        "quand",
        "ou",
        "où",
        "et",
        "en",
        "au",
        "aux",
        "ce",
        "c",
        "cette",
        "ces",
        "mon",
        "ma",
        "mes",
        "notre",
        "nos",
        "votre",
        "vos",
        "il",
        "elle",
        "on",
        "nous",
        "vous",
        "ils",
        "elles",
        "se",
        "s",
        "y",
        "a",
        "ne",
        "pas",
        "peut",
        "peuvent",
        "fait",
        "faire",
        "est-ce",
        "today",
        "now",
        "just",
        "really",
        "also",
        "get",
        "got",
    ]
)
SUFFIXES = ("ing", "ed", "es", "s", "e")


@dataclass(frozen=True)
class Card:
    """One knowledge card."""

    id: str
    question: dict[str, str]
    answer: dict[str, str]
    keywords: tuple[str, ...]
    references: tuple[str, ...]
    source: str

    def text(self, field: str, language: str) -> str:
        """Return a field in the language (English if missing).

        Parameters
        ----------
        field : {"question", "answer"}
            Field.
        language : {"en", "fr"}
            Language.

        Returns
        -------
        str
            The text.
        """
        values: dict[str, str] = getattr(self, field)
        return values.get(language) or values["en"]


def tokens(text: str) -> list[str]:
    """Split text into search terms: lower case, no accents, no stop words, stemmed.

    Parameters
    ----------
    text : str
        Text in English or French.

    Returns
    -------
    list of str
        Terms.

    Examples
    --------
    >>> tokens("What are Content Credentials (C2PA)?")
    ['content', 'credential', 'c2pa']
    >>> tokens("Qu’est-ce qu’un hypertrucage ?")
    ['hypertrucag']
    """
    folded = unicodedata.normalize("NFKD", text.lower().replace("’", "'"))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    out = []
    for word in re.findall(r"[a-z0-9]+", folded):
        if word in STOP or len(word) < 2:  # noqa: PLR2004 - single letters carry no meaning here
            continue
        out.append(stem(word))
    return out


def stem(word: str) -> str:
    """Strip one common ending so that "shared", "sharing" and "share" match.

    Parameters
    ----------
    word : str
        A lower-case word.

    Returns
    -------
    str
        The stem (unchanged when it would be shorter than four letters).

    Examples
    --------
    >>> stem("sharing"), stem("shared"), stem("share"), stem("uploaded")
    ('shar', 'shar', 'shar', 'upload')
    """
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4 and not word.endswith("ss"):  # noqa: PLR2004
            return word[: -len(suffix)]
    return word


@cache
def load_cards() -> tuple[Card, ...]:
    """Load the knowledge cards shipped with the package (cached).

    Returns
    -------
    tuple of Card
        Every card, in file order.
    """
    raw: dict[str, Any] = yaml.safe_load(
        files("marscope.explain").joinpath("knowledge.yaml").read_text(encoding="utf-8")
    )
    return tuple(
        Card(
            id=key,
            question=dict(card["q"]),
            answer=dict(card["answer"]),
            keywords=tuple(card.get("keywords", [])),
            references=tuple(card.get("references", [])),
            source=str(card.get("source", "")),
        )
        for key, card in raw["cards"].items()
    )


@cache
def _anchors() -> list[tuple[list[set[str]], set[str]]]:
    """Per card: its question terms per language, and all question and keyword terms."""
    out = []
    for card in load_cards():
        per_lang = [set(tokens(card.text("question", lang))) for lang in ("en", "fr")]
        out.append((per_lang, set().union(*per_lang) | {t for keyword in card.keywords for t in tokens(keyword)}))
    return out


@cache
def _index() -> tuple[list[Counter[str]], dict[str, float], float]:
    docs: list[Counter[str]] = []
    for card in load_cards():
        terms: Counter[str] = Counter()
        for lang in ("en", "fr"):
            for term in tokens(card.text("question", lang)):
                terms[term] += FIELD_WEIGHTS["q"]
            for term in tokens(card.text("answer", lang)):
                terms[term] += FIELD_WEIGHTS["answer"]
        for keyword in card.keywords:
            for term in tokens(keyword):
                terms[term] += FIELD_WEIGHTS["keywords"]
        docs.append(terms)
    n = len(docs)
    df = Counter(term for doc in docs for term in doc)
    idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}
    avg = sum(sum(doc.values()) for doc in docs) / max(n, 1)
    return docs, idf, avg


def search(question: str, k: int = 3) -> list[tuple[Card, float]]:
    """Find the cards closest to a question (English or French), best first.

    Parameters
    ----------
    question : str
        The question.
    k : int
        Most cards returned.

    Returns
    -------
    list of (Card, float)
        Cards with a positive score, and the score. ``MATCH`` or more is a close match;
        a card whose question and keywords share no term with the question stays below
        ``MATCH`` whatever its answer says.

    Examples
    --------
    >>> search("What is a deepfake?")[0][0].id
    'deepfake'
    """
    docs, idf, avg = _index()
    query = tokens(question)
    k1, b = 1.5, 0.75
    scored = []
    asked = set(query)
    for card, doc, (q_sets, anchor) in zip(load_cards(), docs, _anchors(), strict=True):
        length = sum(doc.values())
        score = 0.0
        for term in query:
            tf = doc.get(term, 0)
            if tf:
                score += idf[term] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
        if asked:  # how closely it asks the card's own question (1 when the same terms)
            score += MATCH * max(len(asked & q) / len(asked | q) for q in q_sets if q)
        if not anchor & asked:
            score = min(score, MATCH * 0.9)
        if score > 0:
            scored.append((card, round(score, 3)))
    scored.sort(key=lambda pair: -pair[1])
    return scored[:k]
