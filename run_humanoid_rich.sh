#!/bin/bash
source "$HOME/venvs/mjt-env/bin/activate"
cd "$HOME/mujoco-torch"
git fetch origin 2>&1
git pull origin direct-optim-humanoid 2>&1
pip install -e ".[zoo]" -q 2>&1 | tail -3

# Set WANDB_API_KEY in your environment before running
export WANDB_API_KEY="${WANDB_API_KEY:?Set WANDB_API_KEY before running}"

echo "Launching 4 humanoid_rich experiments..."

# GPUs 0+4: PPO (2-GPU: env on 0, training on 4)
CUDA_VISIBLE_DEVICES=0,4 python -u examples/train_ppo.py \
    --env humanoid_rich --num_envs 8192 --frame_skip 5 \
    --frames_per_batch 8192000 --total_frames 500000000 \
    --lr 3e-4 --num_epochs 10 --mini_batch_size 2048 \
    --device cuda:0 --train_device cuda:1 --compile \
    --eval_interval 10 --log_interval 1 \
    --wandb_project mujoco-torch-zoo --seed 42 \
    > /root/ppo_humanoid_rich.log 2>&1 &
echo "PPO humanoid_rich PID=$!"

# GPUs 1+5: SAC (2-GPU: env on 1, training on 5)
CUDA_VISIBLE_DEVICES=1,5 python -u examples/train_sac.py \
    --env humanoid_rich --num_envs 2048 --frame_skip 5 \
    --frames_per_batch 2048000 --total_frames 500000000 \
    --learning_starts 25000 \
    --batch_size 8192 --buffer_size 1000000 --utd_ratio 1 \
    --device cuda:0 --train_device cuda:1 --compile \
    --eval_interval 10 --log_interval 1 \
    --wandb_project mujoco-torch-zoo --seed 42 \
    > /root/sac_humanoid_rich.log 2>&1 &
echo "SAC humanoid_rich PID=$!"

# GPU 2: Direct gradient (adaptive integration)
CUDA_VISIBLE_DEVICES=2 python -u examples/train_direct_humanoid_rich.py \
    --device cuda --num_envs 128 --horizon 5 --frame_skip 2 \
    --lr 3e-4 --num_iters 5000 --grad_clip 1.0 \
    --batchnorm --smooth_collisions --cfd --adaptive_integration \
    --eval_interval 50 --log_interval 10 \
    --wandb_project mujoco-torch-zoo --seed 42 \
    > /root/direct_humanoid_rich.log 2>&1 &
echo "Direct humanoid_rich PID=$!"

# GPU 3: SHAC (adaptive integration)
CUDA_VISIBLE_DEVICES=3 python -u examples/train_shac_humanoid_rich.py \
    --device cuda --num_envs 128 --horizon 5 --frame_skip 2 \
    --lr_actor 3e-4 --lr_critic 1e-3 --gamma 0.99 --tau 0.005 \
    --grad_clip 1.0 --num_iters 5000 \
    --batchnorm --smooth_collisions --cfd --adaptive_integration \
    --eval_interval 50 --log_interval 10 \
    --wandb_project mujoco-torch-zoo --seed 42 \
    > /root/shac_humanoid_rich.log 2>&1 &
echo "SHAC humanoid_rich PID=$!"

echo "Waiting..."
wait
echo "Done."
