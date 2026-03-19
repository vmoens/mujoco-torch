#!/usr/bin/env python3
"""Produce demo videos of both satellite CMG environments.

A sinusoidal gimbal-rate policy drives visible tumbling so the gyroscopic
coupling between CMG gimbals and satellite attitude is clearly visible.

Usage (from the repo root):
    source .venv/bin/activate
    python examples/satellite_video.py
"""

import math

import imageio
import mujoco
import torch
from tensordict import TensorDict

import mujoco_torch
from mujoco_torch.zoo import SatelliteLargeEnv, SatelliteSmallEnv

STEPS = 750
WIDTH, HEIGHT = 640, 480
FPS = 50
torch.set_default_dtype(torch.float64)


def _sinusoidal_action(step, n_gimbals, dt):
    """Slowly-varying sinusoidal gimbal commands that produce sustained torque."""
    t = step * dt
    freqs = [0.6 + 0.35 * i for i in range(n_gimbals)]
    phases = [i * math.pi / n_gimbals for i in range(n_gimbals)]
    return torch.tensor(
        [[0.9 * math.sin(2 * math.pi * f * t + p) for f, p in zip(freqs, phases)]],
    )


def make_video(env_cls, filename, camera_distance, camera_elevation=-20, camera_azimuth=135):
    env = env_cls(num_envs=1)
    env.reset()
    n_gimbals = env.action_spec.shape[-1]
    dt = env._dt

    m = env._m_mj
    m.vis.global_.offwidth = max(m.vis.global_.offwidth, WIDTH)
    m.vis.global_.offheight = max(m.vis.global_.offheight, HEIGHT)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=HEIGHT, width=WIDTH)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = camera_distance
    camera.elevation = camera_elevation
    camera.azimuth = camera_azimuth
    camera.lookat[:] = [0.0, 0.0, 0.0]

    frames = []
    for step in range(STEPS):
        action = _sinusoidal_action(step, n_gimbals, dt)
        td = TensorDict({"action": action}, batch_size=env.batch_size)
        env.step(td)

        mujoco_torch.device_get_into(d, env._dx[0])
        renderer.update_scene(d, camera=camera)
        frames.append(renderer.render().copy())

    imageio.mimwrite(filename, frames, fps=FPS)
    print(f"Saved {filename}  ({len(frames)} frames, {len(frames) / FPS:.1f}s)")


print("Rendering large satellite (4 CMGs, pyramid)...")
make_video(SatelliteLargeEnv, "satellite_large_demo.mp4", camera_distance=5.0)

print("Rendering small satellite (6 CMGs, redundant)...")
make_video(SatelliteSmallEnv, "satellite_small_demo.mp4", camera_distance=0.7)
