#!/usr/bin/env python3
"""SHAC (Short-Horizon Actor-Critic) loss module.

Combines short-horizon differentiable rollouts (physics gradients through
the actor) with a learned value function for terminal bootstrapping and
entropy-regularised stochastic policy for exploration.

Reference: Xu et al., "Accelerated Policy Learning with Parallel
Differentiable Simulation", ICLR 2022.
"""

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict


class SHACLoss(nn.Module):
    """SHAC loss: differentiable actor + value bootstrap + auto-tuned entropy.

    Unlike TorchRL ``LossModule``, this is a plain ``nn.Module`` because the
    actor loss *requires* physics gradients flowing through ``env.step()``,
    and ``LossModule.forward()`` auto-wraps with ``DETERMINISTIC`` exploration
    which would break that.

    Args:
        actor_network: ``ProbabilisticActor`` (TanhNormal, ``return_log_prob=True``).
        value_network: ``ValueOperator`` V(s).
        gamma: Discount factor.
        tau: EMA coefficient for target value network updates.
        target_entropy: Target entropy for alpha tuning (default: ``-act_dim``).
        act_dim: Action dimensionality (used to set default ``target_entropy``).
    """

    def __init__(
        self,
        actor_network,
        value_network,
        *,
        gamma: float = 0.99,
        tau: float = 0.005,
        target_entropy: float | None = None,
        act_dim: int | None = None,
    ):
        super().__init__()
        self.actor_network = actor_network
        self.value_network = value_network
        self.target_value_network = copy.deepcopy(value_network)
        # Freeze target params
        for p in self.target_value_network.parameters():
            p.requires_grad_(False)

        self.gamma = gamma
        self.tau = tau

        if target_entropy is None:
            if act_dim is None:
                raise ValueError("Provide target_entropy or act_dim")
            target_entropy = -float(act_dim)
        self.target_entropy = target_entropy

        self.log_alpha = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ------------------------------------------------------------------
    # Actor loss — backprops through physics
    # ------------------------------------------------------------------

    def actor_loss(self, rollout_td: TensorDict):
        """Compute the actor loss from a differentiable rollout.

        Args:
            rollout_td: ``TensorDict[T, B, ...]`` with keys
                ``("next", "reward")``, ``("sample_log_prob",)``,
                and the final next-state observation at ``rollout_td[-1]["next"]``.

        Returns:
            ``(loss, mean_log_prob)`` — ``loss`` has physics gradients;
            ``mean_log_prob`` is detached for the alpha loss.
        """
        T = rollout_td.shape[0]  # noqa: N806
        rewards = rollout_td["next", "reward"]  # [T, B, 1]
        log_probs = rollout_td["sample_log_prob"]  # [T, B]

        # Discounted return: sum_{t=0}^{T-1} gamma^t * r_t
        discounts = torch.tensor(
            [self.gamma**t for t in range(T)],
            device=rewards.device,
            dtype=rewards.dtype,
        )
        discounted_rewards = (discounts.view(T, 1, 1) * rewards).sum(0)  # [B, 1]

        # Terminal bootstrap: gamma^T * V_target(s_T)  (no actor grads)
        final_obs = rollout_td[-1]["next"]
        with torch.no_grad():
            v_terminal = self.target_value_network(final_obs)["state_value"]  # [B, 1]
        bootstrap = (self.gamma**T) * v_terminal

        # Entropy bonus: -alpha * mean(log_prob) per step
        mean_log_prob = log_probs.mean()
        entropy_bonus = -self.alpha.detach() * mean_log_prob

        returns = discounted_rewards + bootstrap  # [B, 1]
        loss = -(returns.mean() + entropy_bonus)

        return loss, mean_log_prob.detach()

    # ------------------------------------------------------------------
    # Critic loss — no physics gradients
    # ------------------------------------------------------------------

    def critic_loss(self, rollout_td: TensorDict):
        """Compute the critic (value) loss from a rollout.

        Uses n-step return targets computed backwards, all detached.

        Args:
            rollout_td: Same as ``actor_loss``.

        Returns:
            Scalar MSE loss.
        """
        T = rollout_td.shape[0]  # noqa: N806

        with torch.no_grad():
            # Bootstrap from terminal state
            final_obs = rollout_td[-1]["next"]
            g = self.target_value_network(final_obs)["state_value"]  # [B, 1]

            rewards = rollout_td["next", "reward"]  # [T, B, 1]

            # Build targets backwards: G_t = r_t + gamma * G_{t+1}
            targets = []
            for t in reversed(range(T)):
                g = rewards[t] + self.gamma * g
                targets.append(g)
            targets.reverse()
            targets = torch.stack(targets, 0)  # [T, B, 1]

        # Predict V(s_t) for all visited states
        # Flatten [T, B] -> [T*B] for the value network
        obs_td = rollout_td.select("observation").reshape(-1)
        v_pred = self.value_network(obs_td)["state_value"]  # [T*B, 1]
        v_pred = v_pred.view(T, -1, 1)

        loss = 0.5 * (v_pred - targets).pow(2).mean()
        return loss

    # ------------------------------------------------------------------
    # Alpha (entropy temperature) loss
    # ------------------------------------------------------------------

    def alpha_loss(self, mean_log_prob: torch.Tensor):
        """Auto-tune entropy temperature.

        Args:
            mean_log_prob: Detached mean log-probability from ``actor_loss``.
        """
        loss = -(self.log_alpha * (mean_log_prob + self.target_entropy).detach())
        # Clamp log_alpha to prevent runaway entropy scaling
        self.log_alpha.data.clamp_(-5.0, 2.0)
        return loss

    # ------------------------------------------------------------------
    # Target network EMA update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_target(self):
        """Polyak-average update of the target value network."""
        for p, p_tgt in zip(
            self.value_network.parameters(),
            self.target_value_network.parameters(),
        ):
            p_tgt.data.lerp_(p.data, self.tau)
