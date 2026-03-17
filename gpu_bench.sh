#!/bin/bash
set -e

source /root/code/.venv/bin/activate

# Ensure cuDNN is on LD_LIBRARY_PATH for CUDA operations
CUDNN_LIB=$(find /root/code/.venv -name "libcudnn.so*" -type f 2>/dev/null | head -1)
if [ -n "$CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="$(dirname $CUDNN_LIB):${LD_LIBRARY_PATH:-}"
fi

echo "=== Environment ==="
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "import tensordict; print('tensordict', tensordict.__version__)" 2>/dev/null || echo "tensordict version check failed"
python -c "import jax; print('jax', jax.__version__, 'devices:', jax.devices())" 2>/dev/null || echo "JAX not installed"

echo ""
echo "=== Running GPU benchmark ==="
cd /root/mujoco-torch

MODEL="${1:-humanoid}"
CUDA_VISIBLE_DEVICES=0 python -u gpu_bench.py \
    --model "$MODEL" \
    --batch-sizes 1 128 1024 4096 \
    --nsteps 1000
