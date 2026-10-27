"""Deletion edits: frame removal keeps a prefix, patch blanking stays in its square."""

from __future__ import annotations

import pytest
import torch

from mars.eval.faithfulness.deletion import blank_patches, remove_frames


def clip(n_frames: int = 6) -> torch.Tensor:
    """Frames whose value is their index, so moves are visible."""
    return torch.arange(n_frames, dtype=torch.float32)[:, None, None, None].expand(n_frames, 3, 8, 8).clone()


def test_remove_frames_repacks_the_rest_in_order() -> None:
    frames, valid = clip(), torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.bool)
    out, new_valid = remove_frames(frames, valid, torch.tensor([1, 3]))
    assert new_valid.tolist() == [True, True, True, False, False, False]
    assert out[:, 0, 0, 0].tolist() == [0, 2, 4, 4, 4, 4]


def test_remove_frames_refuses_to_empty_a_clip() -> None:
    with pytest.raises(ValueError, match="every genuine frame"):
        remove_frames(clip(2), torch.ones(2, dtype=torch.bool), torch.tensor([0, 1]))


def test_blank_patches_fills_only_the_chosen_squares_with_the_frame_mean() -> None:
    frames = torch.rand(2, 3, 8, 8)
    out = blank_patches(frames, torch.tensor([3]), grid_side=2)  # bottom-right 4 x 4 square
    mean = frames.mean(dim=(2, 3), keepdim=True)
    torch.testing.assert_close(out[:, :, 4:, 4:], mean.expand(2, 3, 4, 4))
    torch.testing.assert_close(out[:, :, :4], frames[:, :, :4])
    torch.testing.assert_close(out[:, :, 4:, :4], frames[:, :, 4:, :4])


def test_choose_frames_ranked_random_and_oracle() -> None:
    from mars.eval.faithfulness.deletion import choose_frames

    scores = torch.tensor([0.1, 0.9, 0.5, 0.8, 0.0, 0.3])
    labels = torch.tensor([0, 1, 1, 1, 0, 0])
    g = torch.Generator().manual_seed(0)
    assert choose_frames(scores, labels, 2, "ranked", g).tolist() == [1, 3]
    assert sorted(choose_frames(scores, labels, 3, "oracle", g).tolist()) == [1, 2, 3]
    assert set(choose_frames(scores, labels, 2, "oracle", g).tolist()) <= {1, 2, 3}
    assert len(set(choose_frames(scores, labels, 4, "random", g).tolist())) == 4
    with pytest.raises(ValueError, match="unknown"):
        choose_frames(scores, labels, 1, "best", g)
