"""Shared pytest fixtures for the mozzavid Test_Suite (Requirement 25.2)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def ij():
    """Return the headless ImageJ gateway shared by the whole test session.

    Session scope pays the 15-20 s JVM start once. Errors from `fiji.get_ij()` are
    deliberately not caught: a broken Fiji install must show up as test errors, not as
    skips that hide it (Requirement 25.7). `mozzavid.fiji` is imported here rather than at
    module level so tests that do not use this fixture are collected even if the JVM
    stack (imagej/scyjava) cannot be imported.
    """
    from mozzavid import fiji

    return fiji.get_ij()


@pytest.fixture
def rng():
    """Return a numpy Generator seeded with 0.

    Function scope gives every test a fresh generator, so a test's random draws do not
    depend on which other tests ran before it.
    """
    return np.random.default_rng(0)
