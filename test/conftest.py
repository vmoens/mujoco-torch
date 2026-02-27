"""Shared pytest configuration and fixtures for mujoco-torch tests."""

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
