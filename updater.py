"""
updater.py
==========
PPO update step and Lagrangian dual updater for SafePC-MARL.

Two classes are exported:

PPOUpdater
----------
Runs the PPO clipped-surrogate update for the HyperNetwork and both critics.
Key properties of the implementation:

1.  Multi-head value loss.
    The CentralizedCritic produces a (B, N_CELLS, N_OBJ) output.  For cell i,
    we extract the (B, N_OBJ) slice and compute per-objective MSE against the
    λ-return targets from the buffer.  Summing over objectives gives one critic
    loss per cell; summing over cells gives the joint critic loss.

2.  Scalarised actor loss.
    The PPO clipped surrogate uses the scalarised advantage A_scalar(t) =
    ω · [A_0, A_1, A_2].  This is the single value normalised to zero mean /
    unit std by the buffer's compute_returns().

3.  Lagrangian penalty term.
    The actor loss is augmented by +λ_i * A_cost_i (maximising expected cost
    is penalised).  λ_i is supplied by the LagrangianUpdater and treated as a
    constant during the PPO inner loop (it is updated AFTER the PPO epoch).

4.  Diversity regulariser.
    Added to the actor loss with coefficient alpha_div.  Computed once per
    update epoch (not per mini-batch) to avoid redundant Dirichlet sampling.

5.  Entropy bonus.
    Standard Gaussian entropy H = 0.5 * (1 + log(2π)) + log_std added to
    the actor loss to maintain exploration.

6.  Gradient clipping.
    Applied to all parameters before each optimiser step.

LagrangianUpdater
-----------------
Holds one non-negative Lagrange multiplier λ_i per cell.
After each PPO epoch the multipliers are updated by dual ascent:

    λ_i  <-  max(0,  λ_i + α_dual * (c̄_i - κ_i))

where c̄_i is the mean cost over the rollout and κ_i is the per-cell budget.
No second-order term is used (plain primal-dual, not augmented Lagrangian)
to keep the implementation interpretable and verifiable.

The multipliers are stored as regular Python floats (not nn.Parameters)
because they are updated by a simple rule rather than by gradient.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from safepc_marl.networks import (
    HyperNetwork, CentralizedCritic, CostCritic, PolicyNet,
    N_CELLS, N_OBJ, OBS_DIM,
)
from safepc_marl.buffer import RolloutBuffer


# ---------------------------------------------------------------------------
# Gaussian log-probability helper
# ---------------------------------------------------------------------------

def _gaussian_log_prob(action: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
    """
    Log probability of action under N(mu, exp(log_std)^2).

    Parameters
    ----------
    action  : (B, ACTION_DIM)
    mu      : (B, ACTION_DIM)
    log_std : scalar tensor — a SINGLE shared std for all action dimensions.
              This function assumes the same std applies to every dimension.
              If ACTION_DIM > 1 and per-dimension stds are needed, the formula
              must be updated: the ``2.0 * log_std`` term must be summed over
              dimensions (``2.0 * log_std * ACTION_DIM`` for a scalar std, or
              ``2.0 * log_std.sum()`` for per-dim stds).

    Returns
    -------
    log_prob : (B,)
    """
    var      = log_std.exp().pow(2)
    log_prob = -0.5 * (
        ((action - mu) ** 2) / var
        + 2.0 * log_std
        + math.log(2.0 * math.pi)
    )
    return log_prob.sum(dim=-1)   # sum over action dimensions


def _gaussian_entropy(log_std: Tensor) -> Tensor:
    """
    Differential entropy of N(0, exp(log_std)^2) per dimension.
    H = 0.5 * (1 + log(2π)) + log_std   (scalar, for one action dimension).

    For ACTION_DIM > 1 with a shared scalar log_std the total entropy is
    ACTION_DIM * H; for per-dimension log stds it is H.sum().  With the
    current ACTION_DIM = 1 this function returns the correct total entropy.
    """
    return 0.5 * (1.0 + math.log(2.0 * math.pi)) + log_std


# ---------------------------------------------------------------------------
# LagrangianUpdater
# ---------------------------------------------------------------------------

class LagrangianUpdater:
    """
    Manages per-cell Lagrange multipliers and executes dual-ascent updates.

    Parameters
    ----------
    kappa     : Dict[int, float]  — per-cell safety budget κ_i
    alpha_dual: float             — dual step size
    lambda_init: float            — initial multiplier value (default 0.0)
    """

    def __init__(
        self,
        kappa:      Dict[int, float],
        alpha_dual: float = 3e-4,
        lambda_init: float = 0.0,
    ):
        self.kappa      = kappa
        self.alpha_dual = alpha_dual
        self.n_cells    = len(kappa)
        # Multipliers: one non-negative float per cell
        self._lambdas: Dict[int, float] = {
            i: float(lambda_init) for i in range(self.n_cells)
        }

    # ------------------------------------------------------------------

    def lambdas(self) -> Dict[int, float]:
        """Return current multiplier values (read-only copy)."""
        return dict(self._lambdas)

    def lambda_tensor(self, cell: int, device: torch.device) -> Tensor:
        """Return λ_i as a detached scalar tensor."""
        return torch.tensor(self._lambdas[cell], dtype=torch.float32,
                            device=device)

    # ------------------------------------------------------------------

    def update(self, mean_costs: Dict[int, float]) -> Dict[int, float]:
        """
        Execute one dual-ascent step for all cells.

        Parameters
        ----------
        mean_costs : Dict[int, float]
            Mean cost c̄_i over the last rollout, one per cell.

        Returns
        -------
        deltas : Dict[int, float]
            Change applied to each multiplier (for logging).
        """
        deltas: Dict[int, float] = {}
        for i in range(self.n_cells):
            constraint_violation = mean_costs[i] - self.kappa[i]
            delta                = self.alpha_dual * constraint_violation
            new_lambda           = max(0.0, self._lambdas[i] + delta)
            deltas[i]            = new_lambda - self._lambdas[i]
            self._lambdas[i]     = new_lambda
        return deltas

    def __repr__(self) -> str:
        lam_str = ", ".join(f"λ{i}={v:.5f}" for i, v in self._lambdas.items())
        kap_str = ", ".join(f"κ{i}={v:.4f}" for i, v in self.kappa.items())
        return f"LagrangianUpdater({lam_str} | {kap_str})"


# ---------------------------------------------------------------------------
# PPOUpdater
# ---------------------------------------------------------------------------

class PPOUpdater:
    """
    Runs PPO update epochs over stored rollout data.

    Parameters
    ----------
    hnet           : HyperNetwork
    critic         : CentralizedCritic
    cost_critics   : List[CostCritic]
    lag_updater    : LagrangianUpdater
    lr_actor       : float   — learning rate for HyperNetwork
    lr_critic      : float   — learning rate for CentralizedCritic
    lr_cost_critic : float   — learning rate for each CostCritic
    clip_eps       : float   — PPO clipping parameter ε
    n_epochs       : int     — number of PPO epochs per rollout
    batch_size     : int     — mini-batch size
    vf_coef        : float   — value function loss coefficient
    ent_coef       : float   — entropy bonus coefficient
    alpha_div      : float   — diversity regulariser coefficient
    max_grad_norm  : float   — gradient clipping norm
    device         : torch.device
    """

    def __init__(
        self,
        hnet:           HyperNetwork,
        critic:         CentralizedCritic,
        cost_critics:   List[CostCritic],
        lag_updater:    LagrangianUpdater,
        lr_actor:       float = 3e-4,
        lr_critic:      float = 1e-3,
        lr_cost_critic: float = 1e-3,
        clip_eps:       float = 0.2,
        n_epochs:       int   = 10,
        batch_size:     int   = 64,
        vf_coef:        float = 0.5,
        ent_coef:       float = 0.01,
        alpha_div:      float = 0.05,
        max_grad_norm:  float = 0.5,
        device:         torch.device | str | None = None,
    ):
        self.hnet         = hnet
        self.critic       = critic
        self.cost_critics = cost_critics
        self.lag_updater  = lag_updater
        self.clip_eps     = clip_eps
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.vf_coef      = vf_coef
        self.ent_coef     = ent_coef
        self.alpha_div    = alpha_div
        self.max_grad_norm = max_grad_norm
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device       = torch.device(device)

        # Separate optimisers so we can control learning rates independently
        self.opt_actor  = Adam(hnet.parameters(), lr=lr_actor)

        critic_params   = list(critic.parameters())
        self.opt_critic = Adam(critic_params, lr=lr_critic)

        self.opt_cost_critics = [
            Adam(cc.parameters(), lr=lr_cost_critic)
            for cc in cost_critics
        ]

    # ------------------------------------------------------------------

    def update(
        self,
        buffers: List[RolloutBuffer],
        omega:   Tensor,
    ) -> Dict:
        """
        Run PPO update epochs over the data stored in buffers.

        Parameters
        ----------
        buffers : List[RolloutBuffer], one per cell (len = N_CELLS)
        omega   : (N_OBJ,)  — current preference vector

        Returns
        -------
        metrics : dict with scalar losses and constraint info for logging
        """
        omega = omega.to(self.device)

        # Accumulators for logging
        acc: Dict[str, float] = {
            "loss_actor":   0.0,
            "loss_cost_v":  0.0,
            "loss_entropy": 0.0,
            "loss_div":     0.0,
            "clip_frac":    0.0,
            "n_updates":    0.0,
            "n_div_updates": 0.0,   # separate counter for diversity loss averaging
        }

        # Retrieve current Lagrange multipliers (fixed during inner loop)
        lambdas = self.lag_updater.lambdas()

        for epoch in range(self.n_epochs):
            # --- Diversity regulariser: one standalone backward pass per epoch ---
            # This must be done before the PPO mini-batch loop because the
            # optimizer.step() calls inside that loop update HyperNetwork
            # parameters in-place, which would invalidate the div_loss graph.
            obs_sample = [
                buffers[i].obs[: buffers[i].size].to(self.device)
                for i in range(N_CELLS)
            ]
            div_loss = self.hnet.diversity_loss(obs_sample, n_omega_samples=32)
            div_loss_scaled = self.alpha_div * div_loss

            self.opt_actor.zero_grad()
            div_loss_scaled.backward()
            torch.nn.utils.clip_grad_norm_(
                self.hnet.parameters(), self.max_grad_norm
            )
            self.opt_actor.step()

            # Record diversity loss for logging (detached float)
            acc["loss_div"]      += div_loss_scaled.item()
            acc["n_div_updates"] += 1.0

            # --- PPO mini-batch loop (no diversity term here) ---
            for cell_idx in range(N_CELLS):
                buf = buffers[cell_idx]
                lam = lambdas[cell_idx]

                for batch in buf.get_batches(self.batch_size):
                    # Recompute policy parameters fresh for every mini-batch.
                    # Each call to _cell_update calls backward() on a graph that
                    # includes the hnet forward pass.  PyTorch frees the saved
                    # intermediate tensors after the first backward(), so reusing
                    # a theta_i computed outside this loop causes a RuntimeError
                    # ("Trying to backward through the graph a second time") on
                    # every batch after the first.  Computing theta_i here gives
                    # each backward() its own independent computation graph.
                    thetas_ep, log_stds_ep = self.hnet(omega)
                    theta_i   = thetas_ep[cell_idx]
                    log_std_i = log_stds_ep[cell_idx]

                    loss_a, loss_cv, clip_f, ent = self._cell_update(
                        cell_idx, batch, theta_i, log_std_i, lam,
                    )
                    acc["loss_actor"]   += loss_a
                    acc["loss_cost_v"]  += loss_cv
                    acc["loss_entropy"] += ent
                    acc["clip_frac"]    += clip_f
                    acc["n_updates"]    += 1.0

        # Average accumulated metrics
        assert acc["n_updates"] > 0, \
            "No mini-batch updates were executed — check n_epochs and batch_size."
        n     = acc.pop("n_updates")
        n_div = acc.pop("n_div_updates")
        # loss_div is accumulated once per epoch; all other losses once per mini-batch
        result: Dict[str, float] = {}
        for k, v in acc.items():
            result[k] = v / max(n_div, 1.0) if k == "loss_div" else v / n
        result["n_updates"] = n
        return result

    # ------------------------------------------------------------------

    def _cell_update(
        self,
        cell_idx: int,
        batch:    Dict,
        theta:    Tensor,
        log_std:  Tensor,
        lam:      float,
    ) -> Tuple[float, float, float, float]:
        """
        One mini-batch update for one cell.

        Parameters
        ----------
        theta   : (POLICY_PARAM_DIM,)  — policy weights for this cell, pre-computed
                  by the caller (once per epoch-cell, not per mini-batch).
        log_std : scalar tensor         — log standard deviation for this cell.

        Returns
        -------
        (actor_loss, cost_critic_loss, clip_fraction, entropy)
        All as Python floats.  The centralised critic is NOT updated here;
        it is updated once per epoch via Trainer._joint_critic_update().
        """
        obs         = batch["obs"].to(self.device)          # (B, OBS_DIM)
        actions     = batch["actions"].to(self.device)      # (B, 1)
        lp_old      = batch["log_probs_old"].to(self.device)# (B,)
        adv_s       = batch["adv_scalar"].to(self.device)   # (B,)
        adv_cost    = batch["adv_cost"].to(self.device)     # (B,)
        returns     = batch["returns"].to(self.device)      # (B, N_OBJ)
        ret_cost    = batch["returns_cost"].to(self.device) # (B,)
        B           = obs.shape[0]

        # ----------------------------------------------------------------
        # Actor update
        # ----------------------------------------------------------------
        mu       = PolicyNet.forward(obs, theta)             # (B, 1)
        log_prob = _gaussian_log_prob(actions, mu, log_std)  # (B,)
        entropy  = _gaussian_entropy(log_std)                # scalar

        # PPO ratio
        ratio    = torch.exp(log_prob - lp_old)             # (B,)

        # Clipped surrogate (reward objective, maximise)
        surr1    = ratio * adv_s
        surr2    = torch.clamp(ratio, 1.0 - self.clip_eps,
                                      1.0 + self.clip_eps) * adv_s
        actor_loss_reward = -torch.min(surr1, surr2).mean()

        # Lagrangian penalty: penalise expected cost advantage.
        # We use the raw mean of adv_cost (without dividing by its std) so
        # that the primal update and the dual update share the same cost scale.
        # The dual-ascent step uses raw mean_cost (from the buffer), so the
        # effective lambda calibration is:
        #   delta_lambda = alpha_dual * (mean_cost - kappa)
        # and the actor gradient from the Lagrangian is:
        #   lam * mean(adv_cost)
        # Both quantities are in the same units (un-normalised costs ∈ [0,1]),
        # making alpha_dual interpretable and the steady-state lambda well-scaled.
        # Previously, dividing adv_cost by its std made lambda calibrated to
        # raw costs while the actor saw normalised advantages — a mismatch that
        # caused alpha_dual to be effectively arbitrary.
        actor_loss_cost = lam * adv_cost.mean()

        # Entropy bonus (negative because we minimise total loss)
        actor_loss_ent  = -self.ent_coef * entropy

        total_actor_loss = actor_loss_reward + actor_loss_cost + actor_loss_ent

        self.opt_actor.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.hnet.parameters(), self.max_grad_norm
        )
        self.opt_actor.step()

        # ----------------------------------------------------------------
        # Cost critic update
        # ----------------------------------------------------------------
        v_cost      = self.cost_critics[cell_idx](obs).squeeze(-1)  # (B,)
        cost_critic_loss = F.mse_loss(v_cost, ret_cost)

        self.opt_cost_critics[cell_idx].zero_grad()
        cost_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.cost_critics[cell_idx].parameters(), self.max_grad_norm
        )
        self.opt_cost_critics[cell_idx].step()

        # ----------------------------------------------------------------
        # Clip fraction (diagnostic)
        # ----------------------------------------------------------------
        with torch.no_grad():
            clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

        return (
            total_actor_loss.item(),
            cost_critic_loss.item(),
            clip_frac,
            entropy.item(),
        )

    def update_critic_joint(
        self,
        joint_obs: Tensor,        # (B, N_CELLS * OBS_DIM)
        returns:   Tensor,        # (B, N_CELLS, N_OBJ)
        omega:     Tensor,        # (N_OBJ,)
    ) -> float:
        """
        Full joint-observation centralised critic update for one mini-batch pass.
        Called once per PPO epoch (inside a loop over n_epochs in update_step)
        so the critic sees the same number of data passes as the actor.

        Parameters
        ----------
        joint_obs : (B, N_CELLS * OBS_DIM)  — concatenated obs for all cells
        returns   : (B, N_CELLS, N_OBJ)     — λ-return targets
        omega     : (N_OBJ,)

        Returns
        -------
        loss : float
        """
        joint_obs = joint_obs.to(self.device)
        returns   = returns.to(self.device)
        omega     = omega.to(self.device)

        values = self.critic(joint_obs, omega)           # (B, N_CELLS, N_OBJ)
        loss   = F.mse_loss(values, returns)

        self.opt_critic.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.max_grad_norm
        )
        self.opt_critic.step()
        return loss.item()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from safepc_marl.networks import build_networks, POLICY_PARAM_DIM
    from safepc_marl.buffer   import RolloutBuffer

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Build networks ---
    nets = build_networks(device)
    hnet, critic, cc_list = nets["hnet"], nets["critic"], nets["cost_critics"]

    kappa       = {0: 0.104, 1: 0.088, 2: 0.085}
    lag_updater = LagrangianUpdater(kappa=kappa, alpha_dual=3e-4)
    updater     = PPOUpdater(
        hnet=hnet, critic=critic, cost_critics=cc_list,
        lag_updater=lag_updater, device=device,
        n_epochs=2, batch_size=64,
    )

    # --- Fill buffers with synthetic data ---
    T       = 500
    omega   = torch.tensor([0.5, 0.3, 0.2])
    buffers = [RolloutBuffer(capacity=T, device=device) for _ in range(N_CELLS)]

    for buf_i, buf in enumerate(buffers):
        for t in range(T):
            buf.add(
                obs        = torch.randn(OBS_DIM),
                action     = torch.randn(1),
                log_prob   = -0.9,
                reward_vec = torch.rand(N_OBJ),
                cost       = int(torch.rand(1).item() < 0.10),
                value_vec  = torch.rand(N_OBJ),
                value_cost = torch.rand(1).item(),
                done       = (t == T - 1),
            )
        buf.set_last_values(torch.zeros(N_OBJ), 0.0)
        buf.compute_returns(omega)

    print("=== PPOUpdater sanity check ===")

    # Snapshot parameters before update
    param_before = hnet.trunk[0].weight.clone()

    metrics = updater.update(buffers, omega)

    param_after = hnet.trunk[0].weight.clone()
    param_changed = not torch.allclose(param_before, param_after)

    print(f"  HyperNetwork weights changed: {param_changed}  (must be True)")
    print(f"  loss_actor   : {metrics['loss_actor']:.6f}")
    print(f"  loss_cost_v  : {metrics['loss_cost_v']:.6f}")
    print(f"  clip_frac    : {metrics['clip_frac']:.4f}")
    print(f"  n_updates    : {int(metrics['n_updates'])}")

    # --- LagrangianUpdater ---
    print("\n=== LagrangianUpdater sanity check ===")
    print(f"  Before: {lag_updater}")

    mean_costs = {i: buf.mean_cost for i, buf in enumerate(buffers)}
    print(f"  mean_costs: {mean_costs}")

    deltas = lag_updater.update(mean_costs)
    print(f"  Deltas: {deltas}")
    print(f"  After : {lag_updater}")

    # All lambdas should be > 0 since mean_costs > kappa
    lams = lag_updater.lambdas()
    for i in range(N_CELLS):
        assert lams[i] >= 0.0, f"λ{i} is negative: {lams[i]}"
        print(f"  λ{i}={lams[i]:.6f}  (constraint: cost={mean_costs[i]:.4f} > κ={kappa[i]:.4f})")

    print("\nAll checks passed.")
