"""Tests for the package itself."""

import marscope


def test_version_and_exports() -> None:
    assert marscope.__version__
    assert {"MARScope", "VideoResult", "apply_policy"} <= set(marscope.__all__)
