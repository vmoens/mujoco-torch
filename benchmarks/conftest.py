"""Benchmark fixtures: GPU assignment, state cleanup, model/batch parametrization."""

import gc
import os

import pytest
import torch

ALL_MODELS = ["humanoid", "ant", "halfcheetah", "walker2d", "hopper"]
BATCH_SIZES = [32768, 4096, 1024, 128, 1]


# ---------------------------------------------------------------------------
# --parallel flag: activates pytest-xdist with one worker per GPU
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    parser.addoption(
        "--parallel",
        action="store_true",
        default=False,
        help="Run benchmarks in parallel across GPUs (requires pytest-xdist)",
    )


def pytest_configure(config):
    if config.getoption("--parallel", default=False):
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        config.option.numprocesses = n_gpus


# ---------------------------------------------------------------------------
# GPU assignment (one dedicated GPU per xdist worker)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def _gpu_setup(worker_id):
    """Assign a dedicated GPU based on xdist worker ID."""
    torch.set_default_dtype(torch.float64)

    if not torch.cuda.is_available():
        return

    gpu_id = 0
    if worker_id != "master":
        gpu_id = int(worker_id.replace("gw", "")) % torch.cuda.device_count()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    torch.cuda.set_device(0)


# ---------------------------------------------------------------------------
# Per-test cleanup (isolation without subprocess overhead)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset compile caches and free GPU memory between tests."""
    torch._dynamo.reset()
    torch.compiler.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model / batch fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=ALL_MODELS)
def model_name(request):
    return request.param


@pytest.fixture(params=BATCH_SIZES)
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
