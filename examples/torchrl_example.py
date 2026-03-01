#!/usr/bin/env python3
"""TorchRL integration example for mujoco-torch.

Demonstrates wrapping mujoco-torch in a **batched** TorchRL EnvBase with
``batch_size=[num_envs]``.  Each call to ``_step`` advances all environments.

On GPU, the loop can be replaced with ``torch.compile(torch.vmap(step))``
for true batch parallelism — see ``examples/batched_comparison.py``.

Run:
    pip install torchrl
    python examples/torchrl_example.py
"""

import mujoco
import torch
from etils import epath
from tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.utils import check_env_specs

import mujoco_torch

MODEL_XML = (epath.resource_path("mujoco_torch") / "test_data" / "ant.xml").read_text()


class MujocoTorchEnv(EnvBase):
    """A batched TorchRL environment backed by mujoco-torch.

    Args:
        xml_string: MuJoCo XML model string.
        num_envs: number of parallel environments (sets ``batch_size``).
        max_episode_steps: truncation horizon per environment.
        device: torch device.
        dtype: floating-point dtype for observations, actions, rewards.
    """

    def __init__(
        self,
        xml_string: str,
        num_envs: int = 16,
        max_episode_steps: int = 1000,
        device=None,
        dtype=torch.float64,
    ):
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        self.dtype = dtype
        self.num_envs = num_envs
        self.max_episode_steps = max_episode_steps

        m_mj = mujoco.MjModel.from_xml_string(xml_string)
        self._m_mj = m_mj
        self.mx = mujoco_torch.device_put(m_mj)
        if device is not None:
            self.mx = self.mx.to(device)

        nq, nv, nu = m_mj.nq, m_mj.nv, m_mj.nu

        self.observation_spec = Composite(
            qpos=Unbounded(shape=(num_envs, nq), dtype=dtype, device=self.device),
            qvel=Unbounded(shape=(num_envs, nv), dtype=dtype, device=self.device),
            batch_size=[num_envs],
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(num_envs, nu),
            dtype=dtype,
            device=self.device,
        )
        self.reward_spec = Unbounded(shape=(num_envs, 1), dtype=dtype, device=self.device)

        d_mj = mujoco.MjData(m_mj)
        mujoco.mj_forward(m_mj, d_mj)
        self._dx0 = mujoco_torch.device_put(d_mj)
        if device is not None:
            self._dx0 = self._dx0.to(device)

    def _make_batch(self, n):
        return torch.stack([self._dx0.clone() for _ in range(n)])

    def _reset(self, tensordict=None, **kwargs):
        reset_mask = None
        if tensordict is not None and "_reset" in tensordict.keys():
            reset_mask = tensordict["_reset"].squeeze(-1)

        if reset_mask is None or not hasattr(self, "_dx"):
            self._dx = self._make_batch(self.num_envs)
            self._step_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        else:
            n_reset = int(reset_mask.sum())
            if n_reset > 0:
                self._dx[reset_mask] = self._make_batch(n_reset)
                self._step_count[reset_mask] = 0

        return TensorDict(
            {
                "qpos": self._dx.qpos.to(self.dtype),
                "qvel": self._dx.qvel.to(self.dtype),
                "done": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _step(self, tensordict):
        action = tensordict["action"].to(self.dtype)

        results = []
        for i in range(self.num_envs):
            dx_i = self._dx[i].replace(ctrl=action[i])
            results.append(mujoco_torch.step(self.mx, dx_i))
        self._dx = torch.stack(results)
        self._step_count += 1

        ctrl_cost = 0.5 * (action**2).sum(dim=-1, keepdim=True)
        reward = (-ctrl_cost).to(self.dtype)

        terminated = torch.zeros(*self.batch_size, 1, dtype=torch.bool, device=self.device)
        truncated = (self._step_count >= self.max_episode_steps).unsqueeze(-1)
        done = terminated | truncated

        return TensorDict(
            {
                "qpos": self._dx.qpos.to(self.dtype),
                "qvel": self._dx.qvel.to(self.dtype),
                "reward": reward,
                "done": done,
                "terminated": terminated,
            },
            batch_size=self.batch_size,
            device=self.device,
        )

    def _set_seed(self, seed):
        torch.manual_seed(seed)


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    num_envs = 16
    env = MujocoTorchEnv(MODEL_XML, num_envs=num_envs, max_episode_steps=200)
    print(f"Created batched env: {num_envs} parallel envs  (batch_size={env.batch_size})")

    print("Checking env specs ...")
    check_env_specs(env, seed=42)
    print("Env specs OK!\n")

    rollout = env.rollout(max_steps=50)
    total_reward = rollout["next", "reward"].sum(dim=0).squeeze(-1)
    print(f"Rollout: {rollout.shape[-1]} steps x {num_envs} envs")
    print(f"  Mean total reward:  {total_reward.mean():.2f}")
    print(f"  Reward std:         {total_reward.std():.2f}")
    print(f"  Final qpos[0, :3]:  {rollout[-1]['next', 'qpos'][0, :3].tolist()}")
