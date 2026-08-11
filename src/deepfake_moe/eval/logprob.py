"""First-token logprob to per-stream P(fake) helpers."""

from __future__ import annotations

import math
import re
from typing import Any


def bool_token_ids(tokenizer: Any) -> tuple[set[int], set[int]]:
    """Return single-token IDs representing JSON true and false values."""
    true_ids: set[int] = set()
    false_ids: set[int] = set()
    for surface in ("true", " true", "True", " True"):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            true_ids.add(ids[0])
    for surface in ("false", " false", "False", " False"):
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if len(ids) == 1:
            false_ids.add(ids[0])
    return true_ids, false_ids


def _renorm_true_prob(logprob_true: float | None, logprob_false: float | None) -> float | None:
    """Renormalize P(true) over true and false from their logprobs."""
    if logprob_true is None and logprob_false is None:
        return None
    pt = math.exp(logprob_true) if logprob_true is not None else 0.0
    pf = math.exp(logprob_false) if logprob_false is not None else 0.0
    denominator = pt + pf
    if denominator <= 0.0:
        return None
    return pt / denominator


def stream_probs_from_step_logprobs(
    step_logprobs: dict[int, float],
    true_ids: set[int],
    false_ids: set[int],
) -> float | None:
    """Return renormalized P(true) from one decoding step."""
    logprob_true = max((step_logprobs[token_id] for token_id in true_ids if token_id in step_logprobs), default=None)
    logprob_false = max(
        (step_logprobs[token_id] for token_id in false_ids if token_id in step_logprobs),
        default=None,
    )
    return _renorm_true_prob(logprob_true, logprob_false)


def field_value_step_indices(
    token_ids: list[int],
    tokenizer: Any,
    true_ids: set[int],
    false_ids: set[int],
) -> dict[str, int]:
    """Locate boolean value-token steps for the two locked-schema fields."""
    boolean_ids = true_ids | false_ids
    remaining = {"video_fake", "audio_fake"}
    found: dict[str, int] = {}
    for index, token_id in enumerate(token_ids):
        if token_id not in boolean_ids:
            continue
        prefix = tokenizer.decode(token_ids[:index], skip_special_tokens=True)
        for field in tuple(remaining):
            if re.search(rf'"{re.escape(field)}"\s*:\s*$', prefix):
                found[field] = index
                remaining.remove(field)
                break
        if not remaining:
            break
    return found
