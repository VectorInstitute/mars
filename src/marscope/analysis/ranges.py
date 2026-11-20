"""Usual ranges: where a score sits among a model's scores on the benchmark test videos.

Each model ships one ranges file per precision with 101 percentiles of its video score
on 11,098 real and 12,435 fake test videos. These are benchmark videos, not everyday
video: a real phone clip can sit above most benchmark real videos without being fake.
Texts that use a rank say so.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from marscope.schema import ScoreContext


LOW_PERCENTILE = 5
HIGH_PERCENTILE = 95


@dataclass(frozen=True)
class UsualRanges:
    """Percentiles 0 to 100 of a model's score on benchmark real and fake videos."""

    real: np.ndarray
    fake: np.ndarray
    n_real: int
    n_fake: int

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "UsualRanges":
        """Build from the parsed JSON file.

        Parameters
        ----------
        data : dict
            ``{"real": [101 floats], "fake": [101 floats], "n_real": int, "n_fake": int,
            ...}``.

        Returns
        -------
        UsualRanges
            The ranges.

        Raises
        ------
        ValueError
            If a list is not 101 non-decreasing values.
        """
        real = np.asarray(data["real"], dtype=np.float64)
        fake = np.asarray(data["fake"], dtype=np.float64)
        for name, values in (("real", real), ("fake", fake)):
            if values.shape != (101,) or np.any(np.diff(values) < 0):
                raise ValueError(f"{name} must be 101 non-decreasing percentiles")
        return cls(real=real, fake=fake, n_real=int(str(data.get("n_real", 0))), n_fake=int(str(data.get("n_fake", 0))))

    @classmethod
    def from_file(cls, path: str | Path) -> "UsualRanges":
        """Read a ``usual_ranges*.json`` file.

        Parameters
        ----------
        path : str or Path
            The file.

        Returns
        -------
        UsualRanges
            The ranges.
        """
        return cls.from_dict(json.loads(Path(path).read_text()))

    def real_range(self) -> tuple[float, float]:
        """Return the real usual range (5th to 95th percentile).

        Returns
        -------
        tuple of float
            Low and high end.
        """
        return float(self.real[LOW_PERCENTILE]), float(self.real[HIGH_PERCENTILE])

    def fake_range(self) -> tuple[float, float]:
        """Return the fake usual range (5th to 95th percentile).

        Returns
        -------
        tuple of float
            Low and high end.
        """
        return float(self.fake[LOW_PERCENTILE]), float(self.fake[HIGH_PERCENTILE])

    def context(self, score: float) -> ScoreContext:
        """Place a video score in both usual ranges.

        Parameters
        ----------
        score : float
            Video score (mean ``fake_prob`` over clips).

        Returns
        -------
        ScoreContext
            Ranks, both ranges and whether it is a high fake score.
        """
        return ScoreContext(
            real_rank=round(rank(score, self.real), 1),
            fake_rank=round(rank(score, self.fake), 1),
            real_range=_round_pair(self.real_range()),
            fake_range=_round_pair(self.fake_range()),
            high_fake_score=score > self.real[HIGH_PERCENTILE],
        )


def rank(score: float, percentiles: np.ndarray) -> float:
    """Percent of the reference videos that score below ``score``.

    Linear between percentiles, 0 below the minimum and 100 above the maximum. Flat
    stretches (repeated percentile values) give the highest matching percentile.

    Parameters
    ----------
    score : float
        Video score.
    percentiles : numpy.ndarray
        101 non-decreasing values.

    Returns
    -------
    float
        Rank from 0 to 100.

    Examples
    --------
    >>> p = np.linspace(0.0, 1.0, 101)
    >>> rank(0.3, p)
    30.0
    >>> rank(2.0, p)
    100.0
    """
    if score <= percentiles[0]:
        return 0.0
    if score >= percentiles[-1]:
        return 100.0
    hi = int(np.searchsorted(percentiles, score, side="right"))
    lo = hi - 1
    span = percentiles[hi] - percentiles[lo]
    frac = 0.0 if span == 0 else (score - percentiles[lo]) / span
    return float(lo + frac)


REAL_LEAN_RANK = 80.0  # below this real rank on every model, the scores lean real


RARITY = {
    "en": {
        "very": "Very unusual for a real video",
        "unusual": "Unusual for a real video",
        "typical": "Typical for a real video",
        "low": "Low, even for a real video",
        "fewer": "fewer than 1 in 100 real videos score this high{on}",
        "one_in": "about 1 in {n} real videos score this high{on}",
        "many": "many real videos score this high{on}",
        "most": "most real videos score higher{on}",
        "on": " on {m}",
    },
    "fr": {
        "very": "Très inhabituel pour une vraie vidéo",
        "unusual": "Inhabituel pour une vraie vidéo",
        "typical": "Habituel pour une vraie vidéo",
        "low": "Bas, même pour une vraie vidéo",
        "fewer": "moins de 1 vraie vidéo sur 100 obtient un score aussi haut{on}",
        "one_in": "environ 1 vraie vidéo sur {n} obtient un score aussi haut{on}",
        "many": "beaucoup de vraies vidéos obtiennent un score aussi haut{on}",
        "most": "la plupart des vraies vidéos obtiennent un score plus haut{on}",
        "on": " avec {m}",
    },
}


def rarity(real_rank: float, model: str = "", language: str = "en") -> tuple[str, str]:
    """Say how unusual a score is for a real video, in words and odds (no verdict).

    Parameters
    ----------
    real_rank : float
        Percent of the model's real test videos scoring lower (``real_rank``).
    model : str
        Model name for the odds line ("... on MARS"); empty leaves it out.
    language : {"en", "fr"}
        Language.

    Returns
    -------
    tuple of (str, str)
        A label ("Very unusual for a real video") and an odds line ("fewer than 1
        in 100 real videos score this high on MARS").

    Examples
    --------
    >>> rarity(99.6, "MARS")[0]
    'Very unusual for a real video'
    >>> rarity(99.6, "MARS")[1]
    'fewer than 1 in 100 real videos score this high on MARS'
    >>> rarity(96.0)[1]
    'about 1 in 25 real videos score this high'
    >>> rarity(80.0)[1]
    'about 1 in 5 real videos score this high'
    >>> rarity(50.0)[0], rarity(10.0)[0]
    ('Typical for a real video', 'Low, even for a real video')
    """
    t = RARITY.get(language, RARITY["en"])
    on = t["on"].format(m=model) if model else ""
    above = 100.0 - real_rank
    if real_rank >= 95:  # noqa: PLR2004 - the bands are the explanation, written out here
        label = t["very"]
    elif real_rank >= 75:  # noqa: PLR2004
        label = t["unusual"]
    elif real_rank >= 25:  # noqa: PLR2004
        return t["typical"], t["many"].format(on=on)
    else:
        return t["low"], t["most"].format(on=on)
    if above < 1:
        return label, t["fewer"].format(on=on)
    return label, t["one_in"].format(n=int(100 / above + 0.5), on=on)  # half up, as the UI rounds


def _round_pair(pair: tuple[float, float]) -> tuple[float, float]:
    return round(pair[0], 3), round(pair[1], 3)
