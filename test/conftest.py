"""Shared pytest configuration and fixtures for mujoco-torch tests.

NOTE: The tests currently use ``absl.testing.absltest.TestCase`` and
``parameterized.TestCase`` which do not consume pytest fixtures.  The
``device`` fixture below is provided for future pytest-native tests.
Until the existing tests are migrated (or decorated with
``@pytest.mark.usefixtures``), passing ``--device cuda`` registers the
flag but does **not** cause absltest-based tests to run on CUDA.
"""

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run tests on (cpu or cuda)",
    )


@pytest.fixture
def device(request):
    dev = request.config.getoption("--device")
    if dev == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device(dev)
