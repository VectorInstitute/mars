"""Logging setup."""

import logging
import sys


def get_logger(name: str = "deepfake_moe", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent).

    Examples
    --------
    >>> log = get_logger("test")
    >>> log.name
    'test'
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False

    return logger
