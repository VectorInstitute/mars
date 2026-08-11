"""Tests for first-token boolean logprob helpers."""

import math

import pytest

from deepfake_moe.eval.logprob import (
    bool_token_ids,
    field_value_step_indices,
    stream_probs_from_step_logprobs,
)


class _Tokenizer:
    _pieces = {
        1: '{"video_fake":',
        2: " true",
        3: ',"reason":"true story",',
        4: '"audio_fake":',
        5: " false",
    }
    _surfaces = {"true": [2], " true": [2], "false": [5], " false": [5]}

    def encode(self, surface: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return self._surfaces.get(surface, [10, 11])

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "".join(self._pieces[token_id] for token_id in token_ids)


def test_bool_token_ids_keeps_only_single_tokens() -> None:
    true_ids, false_ids = bool_token_ids(_Tokenizer())
    assert true_ids == {2}
    assert false_ids == {5}


def test_stream_prob_renormalizes_boolean_logprobs() -> None:
    prob = stream_probs_from_step_logprobs({2: math.log(0.2), 5: math.log(0.2)}, {2}, {5})
    assert prob == pytest.approx(0.5)
    assert stream_probs_from_step_logprobs({2: -4.0}, {2}, {5}) == 1.0
    assert stream_probs_from_step_logprobs({}, {2}, {5}) is None


def test_field_value_steps_are_key_anchored() -> None:
    steps = field_value_step_indices([1, 2, 3, 4, 5], _Tokenizer(), {2}, {5})
    assert steps == {"video_fake": 1, "audio_fake": 4}
