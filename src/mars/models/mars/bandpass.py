"""Band-pass filters over the patch grid and the frame axis.

A band is the difference of two smoothings, S(a) x - S(b) x for a < b.
Every smoothing matrix has unit row sums, so a band cancels the constant mode.
"""

from __future__ import annotations

import math

import torch


def scale_pairs(scales: list[float]) -> list[tuple[float, float]]:
    """Turn an increasing scale ladder into consecutive (low, high) bands."""
    if len(scales) < 2:
        raise ValueError(f"need at least two scales to form a band, got {scales}")
    if scales[0] <= 0:
        raise ValueError(f"scales must be positive, got {scales}")
    if any(b <= a for a, b in zip(scales, scales[1:])):
        raise ValueError(f"scales must be strictly increasing, got {scales}")
    return list(zip(scales, scales[1:]))


def _path_laplacian(n: int) -> torch.Tensor:
    """Combinatorial Laplacian of an n-node path graph."""
    if n < 2:
        raise ValueError(f"path length must be at least 2, got {n}")
    lap = torch.zeros(n, n, dtype=torch.float64)
    index = torch.arange(n - 1)
    lap[index, index + 1] = -1.0
    lap[index + 1, index] = -1.0
    lap[index, index] += 1.0
    lap[index + 1, index + 1] += 1.0
    return lap


def heat_matrix(n: int, tau: float) -> torch.Tensor:
    """Graph heat kernel exp(-tau L) on an n-node path."""
    if tau < 0:
        raise ValueError(f"tau must be non-negative, got {tau}")
    values, vectors = torch.linalg.eigh(_path_laplacian(n))
    return (vectors * torch.exp(-tau * values)) @ vectors.T


def gaussian_matrix(n: int, sigma: float) -> torch.Tensor:
    """Gaussian smoothing on n positions, rows normalized to sum to one."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    offsets = torch.arange(n, dtype=torch.float64)
    kernel = torch.exp(-0.5 * ((offsets[:, None] - offsets[None, :]) / sigma) ** 2)
    return kernel / kernel.sum(dim=1, keepdim=True)


def smoothing_matrix(n: int, tau: float, kind: str) -> torch.Tensor:
    """Smoothing matrix at diffusion scale tau, Gaussian or heat kernel."""
    if kind == "dog":
        return gaussian_matrix(n, math.sqrt(2.0 * tau))
    if kind == "heat":
        return heat_matrix(n, tau)
    raise ValueError(f"unknown operator kind {kind!r}, expected 'dog' or 'heat'")


def prefix_tables(n_frames: int, tau: float, kind: str) -> torch.Tensor:
    """One smoothing matrix per valid prefix length, stacked as (n+1, n, n).

    Row L smooths the first L frames and leaves the rest as identity, so
    padded slots never bleed into real ones. Row n_frames is the unrestricted
    matrix, which is what makes an unpadded clip match the unmasked path.
    """
    table = torch.eye(n_frames, dtype=torch.float64).repeat(n_frames + 1, 1, 1)
    for length in range(2, n_frames + 1):
        table[length, :length, :length] = smoothing_matrix(length, tau, kind)
    return table


def smooth_grid(x: torch.Tensor, matrix: torch.Tensor, side: int) -> torch.Tensor:
    """Smooth (B, T, side*side, D) tokens along both grid axes.

    Both kernels are separable, so the same one-dimensional matrix applies to
    rows and then to columns.
    """
    batch, frames, tokens, dim = x.shape
    if tokens != side * side:
        raise ValueError(f"{tokens} tokens do not form a {side}x{side} grid")
    grid = x.reshape(batch, frames, side, side, dim)
    matrix = matrix.to(dtype=x.dtype, device=x.device)
    grid = torch.einsum("ij,btjkd->btikd", matrix, grid)
    grid = torch.einsum("kl,btild->btikd", matrix, grid)
    return grid.reshape(batch, frames, tokens, dim)


def smooth_time(x: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Smooth (B, T, N, D) tokens along the frame axis."""
    matrix = matrix.to(dtype=x.dtype, device=x.device)
    return torch.einsum("ts,bsnd->btnd", matrix, x)


def smooth_time_masked(x: torch.Tensor, table: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Smooth along time, each clip using the table row for its frame count."""
    lengths = valid.sum(dim=1).clamp(0, table.shape[0] - 1)
    matrices = table.to(dtype=x.dtype, device=x.device)[lengths]
    return torch.einsum("bts,bsnd->btnd", matrices, x)
