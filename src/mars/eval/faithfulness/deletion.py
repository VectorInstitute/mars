"""Deletion edits for the faithfulness test: remove frames or blank patches.

A frame is removed by dropping it from the clip and re-packing the remaining frames
in order, so the valid frames stay a prefix as MARS expects; the freed slots repeat
the last kept frame. A patch is blanked by filling its square of pixels with the
frame's mean colour, in every frame of the clip.
"""

from __future__ import annotations

import torch


def remove_frames(frames: torch.Tensor, valid: torch.Tensor, drop: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop the frames in drop from one clip and re-pack the rest.

    Parameters
    ----------
    frames : torch.Tensor
        (T, 3, S, S) frames of one clip.
    valid : torch.Tensor
        (T,) mask of genuine frames, a prefix.
    drop : torch.Tensor
        Indices of genuine frames to remove.

    Returns
    -------
    tuple of torch.Tensor
        The re-packed (T, 3, S, S) frames and their (T,) prefix mask.
    """
    n_frames = frames.shape[0]
    keep = torch.ones(n_frames, dtype=torch.bool, device=frames.device)
    keep[drop] = False
    keep &= valid.bool()
    kept = frames[keep]
    if kept.shape[0] == 0:
        raise ValueError("cannot remove every genuine frame")
    pad = kept[-1:].expand(n_frames - kept.shape[0], *kept.shape[1:])
    new_valid = torch.zeros(n_frames, dtype=torch.bool, device=frames.device)
    new_valid[: kept.shape[0]] = True
    return torch.cat([kept, pad]), new_valid


def blank_patches(frames: torch.Tensor, patches: torch.Tensor, grid_side: int) -> torch.Tensor:
    """Fill the given patches of every frame with that frame's mean colour.

    Parameters
    ----------
    frames : torch.Tensor
        (T, 3, S, S) frames of one clip, S divisible by grid_side.
    patches : torch.Tensor
        Patch indices in row-major order on the grid_side x grid_side grid.
    grid_side : int
        Patches per side of the encoder's grid.

    Returns
    -------
    torch.Tensor
        A copy of frames with the patches blanked.
    """
    size = frames.shape[-1] // grid_side
    out = frames.clone()
    mean = frames.mean(dim=(2, 3), keepdim=True)
    for patch in patches.tolist():
        row, col = divmod(int(patch), grid_side)
        out[:, :, row * size : (row + 1) * size, col * size : (col + 1) * size] = mean
    return out


def choose_frames(
    scores: torch.Tensor, labels: torch.Tensor, k: int, kind: str, generator: torch.Generator
) -> torch.Tensor:
    """Pick k frames of one clip to remove.

    Parameters
    ----------
    scores : torch.Tensor
        (T,) frame scores.
    labels : torch.Tensor
        (T,) 1 for manipulated frames.
    k : int
        Frames to pick.
    kind : str
        ranked (highest scores), random, or oracle (manipulated frames, drawn at random
        when there are more than k).
    generator : torch.Generator
        Source of the random choices.

    Returns
    -------
    torch.Tensor
        Indices of the chosen frames.
    """
    if kind == "ranked":
        return scores.argsort(descending=True)[:k]
    if kind == "random":
        return torch.randperm(len(scores), generator=generator)[:k]
    if kind == "oracle":
        manipulated = labels.nonzero().flatten()
        return manipulated[torch.randperm(len(manipulated), generator=generator)[:k]]
    raise ValueError(f"unknown frame choice {kind!r}")
