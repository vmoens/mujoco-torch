#!/usr/bin/env python3
"""TorchRL-idiomatic random-policy video + CSV logging for satellite envs.

Demonstrates the standard TorchRL pipeline:
  - ``from_pixels=True`` for pixel observations
  - ``TransformedEnv`` + ``VideoRecorder`` for video capture
  - ``CSVLogger`` for scalar metric logging and mp4 export

Usage (from the repo root):
    source .venv/bin/activate
    python examples/satellite_video_torchrl.py

Outputs are saved under ``satellite_logs/<env_name>/``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from torchrl.envs import StepCounter, TransformedEnv
from tensordict import TensorDictBase
from torchrl.record import CSVLogger, VideoRecorder

from zoo import SatelliteLargeEnv, SatelliteSmallEnv

STEPS = 500
RENDER_SIZE = 256
FPS = 50

torch.set_default_dtype(torch.float64)

def count_steps(self, td: TensorDictBase):
    s = td["step_count"].float().mean().item()
    if s % 10 == 0:
        print(f"Step count: {s:.0f}")

def run(env_cls, name):
    logger = CSVLogger(
        exp_name=name,
        log_dir="satellite_logs",
        video_format="mp4",
        video_fps=FPS,
    )

    base_env = env_cls(
        num_envs=1,
        from_pixels=True,
        render_width=RENDER_SIZE,
        render_height=RENDER_SIZE,
        frame_skip=10,
    ).append_transform(StepCounter())

    recorder = VideoRecorder(
        logger=logger,
        tag="random_policy",
        in_keys=["pixels"],
        skip=1,
        make_grid=False,
    )
    env = TransformedEnv(base_env, recorder)

    class SteadyPolicy:
        def __init__(self, action: torch.Tensor):
            self.action = action

        def __call__(self, td: TensorDictBase):
            return td.set("action", self.action)

    td = env.rollout(policy=SteadyPolicy(torch.ones(env.action_spec.shape)), max_steps=STEPS, callback=count_steps)
    env.transform.dump()

    rewards = td["next", "reward"].squeeze()
    for i in range(len(rewards)):
        logger.log_scalar("step_reward", rewards[i].item(), step=i)

    mean_r = rewards.mean().item()
    total_r = rewards.sum().item()
    logger.log_scalar("mean_reward", mean_r, step=0)
    logger.log_scalar("total_reward", total_r, step=0)

    print(f"  {name}: mean_reward={mean_r:.4f}, "
          f"total_reward={total_r:.2f}")
    print(f"  Video + CSV saved under satellite_logs/{name}/")

    del env


print("=== Satellite CMG random-policy demo (TorchRL) ===\n")

print("Large satellite (4 CMGs, pyramid):")
run(SatelliteLargeEnv, "satellite_large")

print("\nSmall satellite (6 CMGs, redundant):")
run(SatelliteSmallEnv, "satellite_small")
