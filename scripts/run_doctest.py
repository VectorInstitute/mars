#!/usr/bin/env python3
"""Run doctests in the files pre-commit passes.

Do not use ``python -m doctest``: with no files it tests CPython itself
(DebugRunner on 3.12), and importing our ``logging.py`` shadows stdlib logging.
"""

from __future__ import annotations

import sys
from doctest import NORMALIZE_WHITESPACE, testfile


def main(argv: list[str]) -> int:
    """Return the number of failed examples (0 if none of ``argv`` are .py)."""
    failed = 0
    for path in argv:
        if path.endswith(".py"):
            failed += testfile(
                path,
                module_relative=False,
                optionflags=NORMALIZE_WHITESPACE,
            ).failed
    return failed


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
