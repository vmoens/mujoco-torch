#!/bin/bash
set -ex

source /root/code/.venv/bin/activate

echo "=== Before ==="
python -c "import torch; print('torch', torch.__version__)"
python -c "import tensordict; print('tensordict', tensordict.__version__)" 2>/dev/null || true

echo "=== Installing PyTorch from vmoens/nomerg-sum-prs ==="
cd /root/pytorch
git remote add vmoens https://github.com/vmoens/pytorch.git 2>/dev/null || git remote set-url vmoens https://github.com/vmoens/pytorch.git
git fetch vmoens vmoens/nomerg-sum-prs
git checkout -- .
git checkout FETCH_HEAD
pip install -e . --no-build-isolation 2>&1 | tail -20

echo "=== Installing tensordict from main ==="
pip install git+https://github.com/pytorch/tensordict.git@main

echo "=== Updating mujoco-torch ==="
cd /root/mujoco-torch
git checkout -- .
git fetch origin fix-compiles
git checkout fix-compiles
git pull origin fix-compiles
pip install -e .

echo "=== After ==="
python -c "import torch; print('torch', torch.__version__, torch.__file__)"
python -c "import tensordict; print('tensordict', tensordict.__version__)"
python -c "import mujoco_torch; print('mujoco_torch OK')"
