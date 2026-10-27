"""Dispatch a model name to its detector class."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from mars.models.base import BaseDetector


# model.name -> detector class, imported only when selected.
DETECTORS = {
    "demamba": "mars.models.demamba.detector:DeMambaDetector",
    "dfd_hr": "mars.models.dfd_hr.detector:DFDHRDetector",
    "effort": "mars.models.effort.detector:EffortDetector",
    "forada": "mars.models.forada.detector:ForAdaDetector",
    "gend": "mars.models.gend.detector:GenDDetector",
    "mars": "mars.models.mars.detector:MarsDetector",
    "restrav": "mars.models.restrav.detector:ReStraVDetector",
    "unite": "mars.models.unite.detector:UniteDetector",
}


def build_detector(name: str, raw: dict[str, Any]) -> BaseDetector:
    """Construct the detector registered under name.

    Parameters
    ----------
    name : str
        Model key from model.name or --model.
    raw : dict
        Merged YAML config.

    Returns
    -------
    BaseDetector
        Ready for training or evaluation.
    """
    if name not in DETECTORS:
        raise ValueError(f"unknown model {name!r}; registered: {sorted(DETECTORS)}")
    module, _, cls = DETECTORS[name].partition(":")
    detector: type[BaseDetector] = getattr(import_module(module), cls)
    return detector.from_config(raw)
