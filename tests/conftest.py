"""Test-suite session config.

Importing :mod:`libs.scenes` here ensures the Mitsuba variant is
selected before Sionna RT can be imported by any test, which avoids
the variant-switch-after-import bug where radio materials end up
registered for the wrong variant.
"""

import libs.scenes  # noqa: F401  — import side effect: sets Mitsuba variant
import pytest


def pytest_collection_modifyitems(config, items):
    """Register the ``gpu`` marker so unknown-mark warnings don't fire."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: tests that require Sionna RT (and ideally a GPU)"
    )
