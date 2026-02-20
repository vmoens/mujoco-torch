"""Benchmark fixtures: device parametrization, model loading, batch creation."""

import os

import pytest
import torch

# ---------------------------------------------------------------------------
# Device parametrization
# ---------------------------------------------------------------------------


def _available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available() and os.environ.get("MUJOCO_TORCH_DISABLE_MPS") != "1":
        devices.append("mps")
    return devices


@pytest.fixture(params=_available_devices(), scope="session")
def device(request):
    """Yield each available device string, augmenting MPS errors with a hint."""
    dev = request.param
    if dev == "mps":
        try:
            yield dev
        except Exception as exc:
            raise type(exc)(
                f"{exc}\n\nMPS failure — to disable MPS benchmarks, set MUJOCO_TORCH_DISABLE_MPS=1"
            ) from exc
    else:
        yield dev


# ---------------------------------------------------------------------------
# Model / batch fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=["ant.xml", "humanoid.xml"], scope="session")
def model_name(request):
    return request.param


@pytest.fixture(params=[1, 64, 256, 1024], scope="session")
def batch_size(request):
    return request.param


# ---------------------------------------------------------------------------
# MJX skip logic
# ---------------------------------------------------------------------------

_has_mjx = True
try:
    import jax  # noqa: F401
    from mujoco import mjx  # noqa: F401
except ImportError:
    _has_mjx = False


def pytest_collection_modifyitems(config, items):
    if _has_mjx:
        return
    skip_mjx = pytest.mark.skip(reason="JAX / mujoco.mjx not installed")
    for item in items:
        if "mjx" in item.keywords:
            item.add_marker(skip_mjx)
