#!/bin/bash
set -e

source /root/code/.venv/bin/activate

CUDNN_LIB=$(find /root/code/.venv -name "libcudnn.so*" -type f 2>/dev/null | head -1)
if [ -n "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$(dirname $CUDNN_LIB):${LD_LIBRARY_PATH:-}"
fi

echo "=== Environment ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import tensordict; print('tensordict', tensordict.__version__)" 2>/dev/null || echo "tensordict version check failed"

echo ""
echo "=== Running GPU benchmark ==="
cd /root/mujoco-torch

CUDA_VISIBLE_DEVICES=0 python -u /root/mujoco-torch/gpu_bench.py
