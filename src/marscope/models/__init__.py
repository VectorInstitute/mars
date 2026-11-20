"""The released detectors: registry and ONNX Runtime runners."""

from marscope.models.registry import MODEL_ORDER, MODELS, ModelSpec, get_spec, parse_models


__all__ = ["MODELS", "MODEL_ORDER", "ModelSpec", "get_spec", "parse_models"]
