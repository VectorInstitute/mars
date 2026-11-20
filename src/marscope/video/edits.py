"""What-if edits: change a video the way sharing does, then score it again.

Edits act on each decoded RGB frame before the usual preprocessing, so the model sees
exactly what it would see on a re-shared copy: lower resolution, blur, sensor-like noise
and JPEG compression, in that order (compression last, as on upload).
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Edits:
    """One set of edits.

    Parameters
    ----------
    jpeg : int
        JPEG quality, 10 to 100 (100 = no compression).
    scale : int
        Resolution in percent, 25 to 100 (resized down, then back up).
    blur : float
        Gaussian blur radius in pixels of the original frame, 0 to 3.
    noise : float
        Gaussian noise standard deviation as a fraction of full range, 0 to 0.1.
    """

    jpeg: int = 100
    scale: int = 100
    blur: float = 0.0
    noise: float = 0.0

    def __post_init__(self) -> None:
        """Check the ranges."""
        if not (10 <= self.jpeg <= 100 and 25 <= self.scale <= 100 and 0 <= self.blur <= 3 and 0 <= self.noise <= 0.1):
            raise ValueError("edits out of range: jpeg 10-100, scale 25-100, blur 0-3, noise 0-0.1")

    @property
    def identity(self) -> bool:
        """Whether these edits change nothing.

        Examples
        --------
        >>> Edits().identity, Edits(jpeg=50).identity
        (True, False)
        """
        return self.jpeg == 100 and self.scale == 100 and self.blur == 0 and self.noise == 0

    def as_dict(self) -> dict[str, float]:
        """Return the edits as plain values.

        Returns
        -------
        dict
            ``jpeg``, ``scale``, ``blur`` and ``noise``.
        """
        return {k: float(v) for k, v in asdict(self).items()}


# "What if it gets shared?": original, shared once, shared again, shared many times.
SHARE_STEPS = (
    Edits(),
    Edits(jpeg=75, scale=80),
    Edits(jpeg=55, scale=60, blur=0.5),
    Edits(jpeg=35, scale=40, blur=1.0, noise=0.02),
)


def apply_edits(rgb: np.ndarray, edits: Edits, seed: int = 0, pixel_scale: float = 1.0) -> np.ndarray:
    """Apply edits to one frame.

    Parameters
    ----------
    rgb : numpy.ndarray
        (H, W, 3) uint8 RGB frame.
    edits : Edits
        What to change.
    seed : int
        Noise seed (the frame index, so a frame always gets the same noise).
    pixel_scale : float
        Size of this frame relative to the original (for previews on thumbnails: the
        blur radius is scaled by it).

    Returns
    -------
    numpy.ndarray
        The edited (H, W, 3) uint8 RGB frame.

    Examples
    --------
    >>> frame = np.full((40, 60, 3), 128, dtype=np.uint8)
    >>> apply_edits(frame, Edits()) is frame
    True
    >>> apply_edits(frame, Edits(jpeg=30, scale=50, blur=1, noise=0.02)).shape
    (40, 60, 3)
    """
    if edits.identity:
        return rgb
    h, w = rgb.shape[:2]
    img = Image.fromarray(rgb)
    if edits.scale < 100:
        small = (max(1, round(w * edits.scale / 100)), max(1, round(h * edits.scale / 100)))
        img = img.resize(small, Image.Resampling.BICUBIC).resize((w, h), Image.Resampling.BICUBIC)
    if edits.blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=edits.blur * pixel_scale))
    if edits.noise > 0:
        noise = np.random.default_rng(seed).normal(0.0, edits.noise * 255.0, size=(h, w, 3))
        img = Image.fromarray(np.clip(np.asarray(img, dtype=np.float32) + noise, 0, 255).astype(np.uint8))
    if edits.jpeg < 100:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=edits.jpeg)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
    return np.asarray(img, dtype=np.uint8)
