#!/usr/bin/env bash
# Full multi-env / multi-mode / multi-backend sweep driver for steve 305299.
#
# For each env in ENVS, runs three processes sequentially:
#   1. torch.compile sweep (raw physics)
#   2. MJX jit(vmap) sweep
#   3. TorchRL collector + LazyTensorStorage(ndim=2) replay buffer
#
# Each process iterates the full BATCH_SIZES list in decreasing order.
# Output: three JSONL files (one per mode+backend) with one row per
# (env, batch_size). Results stream live via tee to a log file.

set -u -o pipefail

ENVS=(${ENVS:-humanoid ant halfcheetah walker2d hopper})
BATCH_SIZES=(${BATCH_SIZES:-131072 65536 16384 4096 1024 128 64 16 4 1})

OUT_DIR="${OUT_DIR:-$HOME/bench_all_305299}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

COMPILE_TORCH_OUT="$OUT_DIR/compile_torch.jsonl"
COMPILE_MJX_OUT="$OUT_DIR/compile_mjx.jsonl"
COLLECTOR_TORCH_OUT="$OUT_DIR/collector_torch.jsonl"

source /root/venvs/torchrl-env/bin/activate
cd /root/mujoco-torch
export PYTHONPATH="/root/mujoco-torch${PYTHONPATH:+:$PYTHONPATH}"

echo "=== multi-env sweep on $(hostname) ==="
echo "  envs:         ${ENVS[*]}"
echo "  batch_sizes:  ${BATCH_SIZES[*]}"
echo "  out_dir:      $OUT_DIR"
echo "  log_dir:      $LOG_DIR"
echo "  started at:   $(date '+%Y-%m-%d %H:%M:%S')"
echo

run_block() {
    local env="$1"
    local mode="$2"
    local backend="$3"
    local out="$4"
    local tag="${env}/${mode}/${backend}"
    local log="$LOG_DIR/${env}_${mode}_${backend}.log"

    echo "##### ${tag} #####  $(date '+%H:%M:%S')"
    if python -u examples/bench_all.py \
            --env "$env" --mode "$mode" --backend "$backend" \
            --batch_sizes "${BATCH_SIZES[@]}" \
            --out "$out" 2>&1 | tee "$log"; then
        echo "[${tag}] ok"
    else
        rc=$?
        echo "[${tag}] exit rc=$rc (continuing)"
    fi
    echo
}

for env in "${ENVS[@]}"; do
    echo "############################################################"
    echo "# env=${env}    $(date '+%Y-%m-%d %H:%M:%S')"
    echo "############################################################"
    run_block "$env" "compile"   "torch" "$COMPILE_TORCH_OUT"
    run_block "$env" "compile"   "mjx"   "$COMPILE_MJX_OUT"
    run_block "$env" "collector" "torch" "$COLLECTOR_TORCH_OUT"
done

echo "=== all envs done at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "outputs:"
echo "  compile_torch:   $COMPILE_TORCH_OUT"
echo "  compile_mjx:     $COMPILE_MJX_OUT"
echo "  collector_torch: $COLLECTOR_TORCH_OUT"
