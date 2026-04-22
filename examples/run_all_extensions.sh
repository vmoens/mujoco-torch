#!/usr/bin/env bash
# Extension sweep: runs the three variants not covered by run_all_305299.sh.
#
# For each env in ENVS, runs three extra processes sequentially:
#   1. torch.compile (tuned) — coordinate-descent tuning + aggressive fusion
#   2. torch.vmap (eager)    — no torch.compile
#   3. MuJoCo C (CPU, seq)   — stock mujoco.mj_step baseline, B=1 only
#
# Kept as a separate orchestrator so the core 3-variant run_all_305299.sh
# can complete cleanly without mixing cold/warm Inductor caches between
# tuned and untuned compile.

set -u -o pipefail

ENVS=(${ENVS:-humanoid ant halfcheetah walker2d})
BATCH_SIZES=(${BATCH_SIZES:-131072 65536 32768 4096 1024 128 1})
MUJOCO_C_BATCH_SIZES=(${MUJOCO_C_BATCH_SIZES:-1})

OUT_DIR="${OUT_DIR:-$HOME/bench_all_305299}"
LOG_DIR="${LOG_DIR:-$OUT_DIR/logs}"
mkdir -p "$OUT_DIR" "$LOG_DIR"

COMPILE_TUNED_OUT="$OUT_DIR/compile_torch_tuned.jsonl"
VMAP_TORCH_OUT="$OUT_DIR/vmap_torch.jsonl"
COMPILE_MUJOCO_C_OUT="$OUT_DIR/compile_mujoco_c.jsonl"

source /root/venvs/torchrl-env/bin/activate
cd /root/mujoco-torch
export PYTHONPATH="/root/mujoco-torch${PYTHONPATH:+:$PYTHONPATH}"

echo "=== extension sweep on $(hostname) ==="
echo "  envs:         ${ENVS[*]}"
echo "  batch_sizes:  ${BATCH_SIZES[*]}"
echo "  mujoco_c bs:  ${MUJOCO_C_BATCH_SIZES[*]}"
echo "  out_dir:      $OUT_DIR"
echo "  log_dir:      $LOG_DIR"
echo "  started at:   $(date '+%Y-%m-%d %H:%M:%S')"
echo

run_block() {
    local env="$1"
    local mode="$2"
    local backend="$3"
    local out="$4"
    local extra_flag="$5"
    local -a batch_list=("${!6}")
    local tag="${env}/${mode}/${backend}${extra_flag:+/tuned}"
    local log="$LOG_DIR/${env}_${mode}_${backend}${extra_flag:+_tuned}.log"

    echo "##### ${tag} #####  $(date '+%H:%M:%S')"
    if python -u examples/bench_all.py \
            --env "$env" --mode "$mode" --backend "$backend" ${extra_flag} \
            --batch_sizes "${batch_list[@]}" \
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
    run_block "$env" "compile" "torch"    "$COMPILE_TUNED_OUT"    "--tuned" BATCH_SIZES[@]
    run_block "$env" "vmap"    "torch"    "$VMAP_TORCH_OUT"       ""        BATCH_SIZES[@]
    run_block "$env" "compile" "mujoco_c" "$COMPILE_MUJOCO_C_OUT" ""        MUJOCO_C_BATCH_SIZES[@]
done

echo "=== all ext envs done at $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "outputs:"
echo "  compile_tuned:    $COMPILE_TUNED_OUT"
echo "  vmap_torch:       $VMAP_TORCH_OUT"
echo "  compile_mujoco_c: $COMPILE_MUJOCO_C_OUT"
