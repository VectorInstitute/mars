"""Plan which models run together, from free GPU memory and each model's measured peak.

Peaks come from ``serving.json`` (process totals including the CUDA context, so in one
process the real total is lower and the plan is safe). Clips run at batch 1: batch 2
barely improves time per clip.
"""

import logging
import subprocess
from collections.abc import Mapping, Sequence


log = logging.getLogger(__name__)

HEADROOM_GB = 1.0


def plan_waves(peaks_gb: Mapping[str, float], free_gb: float | None, sequential: bool = False) -> list[list[str]]:
    """Group models into waves that fit in free GPU memory, MARS first.

    Parameters
    ----------
    peaks_gb : mapping of str to float
        Model id to peak GiB at batch 1, in the order models should run.
    free_gb : float or None
        Free GPU memory; None (unknown, or CPU) runs one model at a time.
    sequential : bool
        Force one model at a time.

    Returns
    -------
    list of list of str
        Waves; models in a wave run in parallel, waves run one after another.

    Examples
    --------
    >>> peaks = {"mars": 2.93, "gend": 2.25, "dfd-hr": 2.55}
    >>> plan_waves(peaks, 10.0)
    [['mars', 'gend', 'dfd-hr']]
    >>> plan_waves(peaks, 8.0)
    [['mars', 'gend'], ['dfd-hr']]
    >>> plan_waves(peaks, 4.0)
    [['mars'], ['gend'], ['dfd-hr']]
    >>> plan_waves(peaks, None)
    [['mars'], ['gend'], ['dfd-hr']]
    """
    models: Sequence[str] = list(peaks_gb)
    if sequential or free_gb is None:
        return [[m] for m in models]
    waves: list[list[str]] = []
    current: list[str] = []
    used = HEADROOM_GB
    for model in models:
        peak = peaks_gb[model]
        if current and used + peak > free_gb:
            waves.append(current)
            current, used = [], HEADROOM_GB
        current.append(model)
        used += peak
    if current:
        waves.append(current)
    return waves


def free_vram_gb(device_id: int = 0) -> float | None:
    """Free memory on one GPU, from ``nvidia-smi``.

    Parameters
    ----------
    device_id : int
        GPU index.

    Returns
    -------
    float or None
        Free GiB, or None when no NVIDIA GPU can be queried.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={device_id}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
        return float(out.strip().splitlines()[0]) / 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError) as error:
        log.debug("nvidia-smi unavailable: %s", error)
        return None
