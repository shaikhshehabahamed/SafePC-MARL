"""
buffer.py
=========
Rollout buffer for SafePC-MARL.

Stores transitions collected during environment interaction and computes
Generalised Advantage Estimates (GAE-λ) for both the multi-objective reward
signal and the scalar safety cost signal.

One RolloutBuffer instance is created PER CELL.  The trainer holds three of
them (one per cell index 0, 1, 2).

Design
------
All tensors are pre-allocated at construction time for efficiency.
After each rollout, compute_returns() populates advantage and return fields
in-place.  get_batches() then yields randomly ordered mini-batches for PPO.

Multi-objective GAE
-------------------
The CentralizedCritic produces a value estimate of shape (N_OBJ,) for each
(observation, ω) pair.  GAE is run independently for each objective j:

    delta_j(t)  = r_j(t) + gamma * V_j(t+1) * (1 - done(t)) - V_j(t)
    A_j(t)      = delta_j(t) + gamma * gae_lambda * (1-done(t)) * A_j(t+1)

The scalarised advantage used in the PPO objective is then:

    A_scalar(t) = ω · [A_0(t), A_1(t), A_2(t)]

This preserves gradient information for all three objectives without
interference through a shared value head.

Cost GAE
--------
Run separately using the scalar CostCritic, producing A_cost(t).
The dual updater uses the mean of accumulated costs (not advantages) for
the multiplier update; A_cost is used for computing the cost-penalised
value target in the cost critic's regression loss.

Returns (for critic regression)
--------------------------------
    G_j(t)    = A_j(t) + V_j(t)       per objective
    G_cost(t) = A_cost(t) + V_cost(t)
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Generator, Tuple

import numpy as np
import torch
from torch import Tensor

from safepc_marl.networks import OBS_DIM, ACTION_DIM, N_OBJ


# ---------------------------------------------------------------------------
# RolloutBuffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Fixed-capacity rollout buffer for one cell.

    Parameters
    ----------
    capacity  : int   — number of steps to store (equals rollout length T)
    gamma     : float — discount factor
    gae_lambda: float — GAE smoothing parameter λ
    device    : torch.device
    """

    def __init__(
        self,
        capacity:   int,
        gamma:      float = 0.99,
        gae_lambda: float = 0.95,
        device:     torch.device | str | None = None,
    ):
        self.capacity   = capacity
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self._ptr       = 0          # write pointer
        self._full      = False

        # --- Pre-allocate storage ---
        T = capacity
        D = OBS_DIM
        A = ACTION_DIM
        J = N_OBJ

        self.obs        = torch.zeros(T, D,  device=self.device)
        self.actions    = torch.zeros(T, A,  device=self.device)
        self.log_probs  = torch.zeros(T,     device=self.device)
        self.reward_vec = torch.zeros(T, J,  device=self.device)
        self.costs      = torch.zeros(T,     device=self.device)
        self.value_vec  = torch.zeros(T, J,  device=self.device)
        self.value_cost = torch.zeros(T,     device=self.device)
        self.dones      = torch.zeros(T,     device=self.device)

        # Computed by compute_returns()
        self.advantages      = torch.zeros(T, J,  device=self.device)
        self.returns         = torch.zeros(T, J,  device=self.device)
        self.adv_cost        = torch.zeros(T,     device=self.device)
        self.returns_cost    = torch.zeros(T,     device=self.device)
        self.adv_scalar      = torch.zeros(T,     device=self.device)

        # Filled in by compute_returns(); needed for bootstrapping
        self._last_value_vec  = torch.zeros(J,  device=self.device)
        self._last_value_cost = torch.tensor(0.0, device=self.device)

        # State flag: True after compute_returns(), reset to False by reset()
        self._computed: bool = False

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add(
        self,
        obs:        Tensor | np.ndarray,
        action:     Tensor | np.ndarray,
        log_prob:   float | Tensor,
        reward_vec: Tensor | np.ndarray,
        cost:       float | int,
        value_vec:  Tensor | np.ndarray,
        value_cost: float | Tensor,
        done:       bool,
    ) -> None:
        """
        Store one transition.  Writes are circular; raises if buffer is
        already full and compute_returns() has not been called.
        """
        if self._full:
            raise RuntimeError(
                "Buffer is full. Call compute_returns() and reset() before adding."
            )

        i = self._ptr
        self.obs[i]        = self._to_tensor(obs,        (OBS_DIM,))
        self.actions[i]    = self._to_tensor(action,     (ACTION_DIM,))
        self.log_probs[i]  = float(log_prob)
        self.reward_vec[i] = self._to_tensor(reward_vec, (N_OBJ,))
        self.costs[i]      = float(cost)
        self.value_vec[i]  = self._to_tensor(value_vec,  (N_OBJ,))
        self.value_cost[i] = float(value_cost)
        self.dones[i]      = float(done)

        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def set_last_values(
        self,
        last_value_vec:  Tensor | np.ndarray,
        last_value_cost: float | Tensor,
    ) -> None:
        """
        Store bootstrap values for the state immediately after the last
        stored transition.  Must be called before compute_returns().
        """
        self._last_value_vec  = self._to_tensor(last_value_vec, (N_OBJ,))
        self._last_value_cost = torch.tensor(float(last_value_cost),
                                             device=self.device)

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def compute_returns(self, omega: Tensor) -> None:
        """
        Compute GAE advantages and λ-returns for all stored transitions.
        Populates self.advantages, self.returns, self.adv_cost,
        self.returns_cost, and self.adv_scalar in-place.

        Parameters
        ----------
        omega : (N_OBJ,)  — current preference vector (used to scalarise)
        """
        T     = self._ptr          # number of transitions actually stored
        gamma = self.gamma
        lam   = self.gae_lambda

        # --- Multi-objective GAE (backward pass) ---
        # advantages: (T, N_OBJ)
        adv = torch.zeros(T, N_OBJ, device=self.device)
        last_adv = torch.zeros(N_OBJ, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_val  = self._last_value_vec
                # Use the actual done flag stored at this step.  When the
                # episode terminates at step T-1 (the normal case here, since
                # rollout_steps == max_steps_ep), done[T-1] == 1.0 and the
                # bootstrap value is correctly zeroed out.  When T-1 is a
                # mid-episode truncation, done[T-1] == 0.0 and the bootstrap
                # value is used as usual.  Hardcoding 0.0 was wrong: it kept
                # the bootstrap even for terminal states.
                next_done = float(self.dones[T - 1].item())
            else:
                next_val  = self.value_vec[t + 1]
                next_done = float(self.dones[t].item())

            not_done = 1.0 - next_done
            delta    = (self.reward_vec[t]
                        + gamma * next_val * not_done
                        - self.value_vec[t])
            last_adv = delta + gamma * lam * not_done * last_adv
            adv[t]   = last_adv

        self.advantages = adv
        self.returns    = adv + self.value_vec[:T]

        # --- Scalarised advantage for PPO surrogate ---
        # omega: (N_OBJ,); adv: (T, N_OBJ) -> (T,)
        self.adv_scalar = (adv * omega.unsqueeze(0)).sum(dim=-1)

        # --- Cost GAE (backward pass) ---
        adv_cost   = torch.zeros(T, device=self.device)
        last_ac    = torch.tensor(0.0, device=self.device)

        for t in reversed(range(T)):
            if t == T - 1:
                next_vc   = self._last_value_cost
                # Same correction as the multi-objective loop above: read the
                # actual done flag rather than hardcoding 0.0.
                next_done = float(self.dones[T - 1].item())
            else:
                next_vc   = self.value_cost[t + 1]
                next_done = float(self.dones[t].item())

            not_done   = 1.0 - next_done
            delta_c    = (self.costs[t]
                          + gamma * next_vc * not_done
                          - self.value_cost[t])
            last_ac    = delta_c + gamma * lam * not_done * last_ac
            adv_cost[t] = last_ac

        self.adv_cost     = adv_cost
        self.returns_cost = adv_cost + self.value_cost[:T]

        # Normalise scalarised advantage (zero mean, unit std) for stable PPO
        adv_s = self.adv_scalar[:T]
        self.adv_scalar[:T] = (adv_s - adv_s.mean()) / (adv_s.std() + 1e-8)

        # Mark buffer as computed (but not yet reset — get_batches() can run)
        self._computed = True

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def get_batches(
        self,
        batch_size: int,
    ) -> Generator[dict, None, None]:
        """
        Yield randomly shuffled mini-batches from the stored transitions.

        Each yielded dict contains:
            obs, actions, log_probs_old, adv_scalar, returns (N_OBJ),
            adv_cost, returns_cost, costs
        All tensors have a leading batch dimension.

        Parameters
        ----------
        batch_size : int
        """
        T    = self._ptr
        perm = torch.randperm(T, device=self.device)

        for start in range(0, T, batch_size):
            idx = perm[start : start + batch_size]
            yield {
                "obs":           self.obs[idx],
                "actions":       self.actions[idx],
                "log_probs_old": self.log_probs[idx],
                "adv_scalar":    self.adv_scalar[idx],
                "advantages":    self.advantages[idx],       # (B, N_OBJ)
                "returns":       self.returns[idx],          # (B, N_OBJ)
                "adv_cost":      self.adv_cost[idx],
                "returns_cost":  self.returns_cost[idx],
                "costs":         self.costs[idx],
            }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored data. Must be called after each rollout update."""
        self._ptr      = 0
        self._full     = False
        self._computed = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._ptr

    @property
    def mean_cost(self) -> float:
        """Mean cost over stored transitions. Used for dual update."""
        assert self._ptr > 0, "Cannot compute mean_cost on an empty buffer."
        return float(self.costs[:self._ptr].mean().item())

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _to_tensor(self, x, expected_shape: tuple) -> Tensor:
        if isinstance(x, Tensor):
            t = x.detach().to(self.device).float()
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        return t.reshape(expected_shape)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    T      = 500

    buf = RolloutBuffer(capacity=T, gamma=0.99, gae_lambda=0.95, device=device)

    # Simulate a rollout with random data
    for t in range(T):
        obs        = torch.randn(OBS_DIM)
        action     = torch.randn(ACTION_DIM)
        log_prob   = -0.5 * (action ** 2).sum().item()
        reward_vec = torch.rand(N_OBJ)
        cost       = int(torch.rand(1).item() < 0.10)
        value_vec  = torch.rand(N_OBJ)
        value_cost = torch.rand(1).item()
        done       = (t == T - 1)
        buf.add(obs, action, log_prob, reward_vec, cost, value_vec, value_cost, done)

    buf.set_last_values(torch.zeros(N_OBJ), 0.0)
    omega = torch.tensor([0.5, 0.3, 0.2])
    buf.compute_returns(omega)

    print("=== RolloutBuffer sanity check ===")
    print(f"  size             : {buf.size}  (expected {T})")
    print(f"  advantages shape : {buf.advantages.shape}  (expected ({T}, {N_OBJ}))")
    print(f"  returns shape    : {buf.returns.shape}")
    print(f"  adv_scalar shape : {buf.adv_scalar.shape}  (expected ({T},))")
    print(f"  adv_cost shape   : {buf.adv_cost.shape}")
    print(f"  returns_cost     : {buf.returns_cost.shape}")
    print(f"  mean_cost        : {buf.mean_cost:.4f}")

    # Check adv_scalar is normalised
    adv_s = buf.adv_scalar[:T]
    print(f"\n  adv_scalar mean  : {adv_s.mean().item():.6f}  (should be ~0)")
    print(f"  adv_scalar std   : {adv_s.std().item():.6f}   (should be ~1)")

    # Check returns = advantages + values
    diff = (buf.returns - (buf.advantages + buf.value_vec[:T])).abs().max()
    print(f"\n  max |returns - (adv+V)| : {diff.item():.2e}  (should be ~0)")

    diff_c = (buf.returns_cost - (buf.adv_cost + buf.value_cost[:T])).abs().max()
    print(f"  max |ret_cost - (adv_c+V_c)| : {diff_c.item():.2e}  (should be ~0)")

    # Check mini-batch iteration
    n_batches = 0
    n_samples  = 0
    for batch in buf.get_batches(batch_size=64):
        n_batches += 1
        n_samples += batch["obs"].shape[0]
        assert batch["obs"].shape[1] == OBS_DIM
        assert batch["actions"].shape[1] == ACTION_DIM
        assert batch["returns"].shape[1] == N_OBJ
    print(f"\n  Mini-batches (batch=64): {n_batches}  total samples={n_samples}  "
          f"(expected {T} samples, {T // 64} full batches + remainder)")

    print("\nAll checks passed.")
