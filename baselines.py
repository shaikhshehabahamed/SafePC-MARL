"""
baselines.py
============
Four comparison baselines for SafePC-MARL.

Each baseline is self-contained: it has its own actor network, trainer, and
evaluate() method that returns a pd.DataFrame compatible with
Evaluator.compute_hypervolume().

Baseline 1 — PPO-Scalar
    Three independent per-cell MLP actors.  Fixed omega = [1/3, 1/3, 1/3].
    Reward = omega . [r1, r2, r3] (scalarised once, treated as single-obj).
    No safety constraint.

Baseline 2 — MACPO
    Three independent per-cell MLP actors.  Fixed omega = [1/3, 1/3, 1/3].
    Safety constraint enforced via Lagrangian (same dual-ascent rule as
    SafePC-MARL, but applied per-cell with a single fixed omega).
    No hypernetwork; no preference conditioning.

Baseline 3 — CAPQL
    One centralised actor that takes joint observations (N_CELLS*OBS_DIM)
    concatenated with omega as input.  Omega is fixed per run.
    Safety constraint via Lagrangian (one shared multiplier for all cells).
    Represents single-agent safe MORL applied in a centralised manner.

Shared infrastructure
---------------------
All baselines reuse ManufacturingEnv, RolloutBuffer, LagrangianUpdater,
and Evaluator from the main codebase.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam

from safepc_marl.networks  import (
    OBS_DIM, ACTION_DIM, N_CELLS, N_OBJ, OMEGA_DIM,
    POLICY_HIDDEN, CRITIC_HIDDEN,
    _orthogonal_init,
)
from safepc_marl.buffer    import RolloutBuffer
from safepc_marl.updater   import LagrangianUpdater, _gaussian_log_prob, _gaussian_entropy
from safepc_marl.simulator import ManufacturingEnv, VARIANTS
from safepc_marl.evaluator import Evaluator, build_simplex_grid


# ---------------------------------------------------------------------------
# Shared MLP actor (used by all baselines)
# ---------------------------------------------------------------------------

class MLPActor(nn.Module):
    """
    Standard Gaussian MLP actor for one cell (or one centralised agent).

    Architecture: Linear(input_dim, 64), Tanh -> Linear(64, 64), Tanh
                  -> mean head: Linear(64, output_dim), Tanh
    log_std: learnable scalar, initialised to log(0.5).

    Parameters
    ----------
    input_dim  : int  — observation dimension
    output_dim : int  — number of action outputs.  Use ACTION_DIM (=1) for a
                        per-cell actor; use N_CELLS for the CAPQL centralised actor
                        that produces one action per cell simultaneously.
    """

    def __init__(self, input_dim: int = OBS_DIM, output_dim: int = ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, POLICY_HIDDEN), nn.Tanh(),
            nn.Linear(POLICY_HIDDEN, POLICY_HIDDEN), nn.Tanh(),
        )
        self.mean_head = nn.Linear(POLICY_HIDDEN, output_dim)
        self.log_std   = nn.Parameter(torch.tensor(math.log(0.5)))

        _orthogonal_init(self.net, gain=math.sqrt(2))
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

    def forward(self, obs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        mu      : (B, output_dim)  — action mean, values in (−1, 1)
        log_std : scalar tensor
        """
        feat = self.net(obs)
        mu   = torch.tanh(self.mean_head(feat))
        return mu, self.log_std


class MLPCritic(nn.Module):
    """Scalar value function for a single objective."""

    def __init__(self, input_dim: int = OBS_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, CRITIC_HIDDEN), nn.Tanh(),
            nn.Linear(CRITIC_HIDDEN, CRITIC_HIDDEN), nn.Tanh(),
            nn.Linear(CRITIC_HIDDEN, 1),
        )
        _orthogonal_init(self.net, gain=math.sqrt(2))
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs)   # (B, 1)


# ---------------------------------------------------------------------------
# Shared PPO update for MLP-based baselines
# ---------------------------------------------------------------------------

def _ppo_update_no_constraint(
    actor:        MLPActor,
    critic:       MLPCritic,
    opt_actor:    torch.optim.Optimizer,
    opt_critic:   torch.optim.Optimizer,
    buffer:       RolloutBuffer,
    omega:        Tensor,
    device:       torch.device,
    n_epochs:     int   = 10,
    batch_size:   int   = 64,
    clip_eps:     float = 0.2,
    ent_coef:     float = 0.01,
    max_grad_norm:float = 0.5,
) -> Dict:
    """PPO update for an unconstrained baseline (PPO-Scalar)."""
    metrics = dict(loss_actor=0.0, loss_critic=0.0, clip_frac=0.0, n=0.0)

    for _ in range(n_epochs):
        for batch in buffer.get_batches(batch_size):
            obs      = batch["obs"].to(device)
            actions  = batch["actions"].to(device)
            lp_old   = batch["log_probs_old"].to(device)
            adv_s    = batch["adv_scalar"].to(device)
            returns  = batch["returns"].to(device)

            ret_scalar = (returns * omega.unsqueeze(0)).sum(dim=-1)

            mu, log_std = actor(obs)
            lp          = _gaussian_log_prob(actions, mu, log_std)
            # For PPO-Scalar each cell actor produces a single action (action_dim=1).
            # Scale entropy by action_dim for generality (evaluates to 1×H here).
            action_dim  = mu.shape[-1]
            ent         = _gaussian_entropy(log_std) * action_dim
            ratio       = torch.exp(lp - lp_old)
            s1 = ratio * adv_s
            s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_s
            a_loss = -torch.min(s1, s2).mean() - ent_coef * ent

            opt_actor.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_actor.step()

            v      = critic(obs).squeeze(-1)
            v_loss = F.mse_loss(v, ret_scalar)
            opt_critic.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_critic.step()

            metrics["loss_actor"]  += a_loss.item()
            metrics["loss_critic"] += v_loss.item()
            metrics["clip_frac"]   += ((ratio-1).abs() > clip_eps).float().mean().item()
            metrics["n"]           += 1.0

    assert metrics["n"] > 0, "No mini-batch updates; check n_epochs and batch_size."
    n = metrics.pop("n")
    return {k: v / n for k, v in metrics.items()}


def _ppo_update_with_constraint(
    actor:        MLPActor,
    critic:       MLPCritic,
    cost_critic:  MLPCritic,
    opt_actor:    torch.optim.Optimizer,
    opt_critic:   torch.optim.Optimizer,
    opt_cc:       torch.optim.Optimizer,
    buffer:       RolloutBuffer,
    omega:        Tensor,
    lam:          float,
    device:       torch.device,
    n_epochs:     int   = 10,
    batch_size:   int   = 64,
    clip_eps:     float = 0.2,
    ent_coef:     float = 0.01,
    max_grad_norm:float = 0.5,
) -> Dict:
    """PPO update for a constrained baseline (MACPO / CAPQL)."""
    metrics = dict(loss_actor=0.0, loss_critic=0.0, loss_cost_v=0.0,
                   clip_frac=0.0, n=0.0)

    for _ in range(n_epochs):
        for batch in buffer.get_batches(batch_size):
            obs      = batch["obs"].to(device)
            actions  = batch["actions"].to(device)
            lp_old   = batch["log_probs_old"].to(device)
            adv_s    = batch["adv_scalar"].to(device)
            adv_cost = batch["adv_cost"].to(device)
            returns  = batch["returns"].to(device)
            ret_cost = batch["returns_cost"].to(device)

            ret_scalar = (returns * omega.unsqueeze(0)).sum(dim=-1)

            mu, log_std = actor(obs)
            lp          = _gaussian_log_prob(actions, mu, log_std)
            # _gaussian_entropy returns the per-dimension entropy H of N(0, sigma^2).
            # For a multi-dimensional action (CAPQL outputs N_CELLS actions), the
            # total differential entropy is action_dim × H because the dimensions
            # are independent and share the same scalar log_std.  For MACPO
            # (action_dim=1) this evaluates to the standard 1 × H.
            action_dim  = mu.shape[-1]
            ent         = _gaussian_entropy(log_std) * action_dim
            ratio       = torch.exp(lp - lp_old)
            s1 = ratio * adv_s
            s2 = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_s
            # Use raw adv_cost.mean() — no std normalisation — so that the
            # primal update and the dual update share the same cost scale
            # (matching the fix applied to PPOUpdater._cell_update).
            a_loss = -torch.min(s1, s2).mean() \
                     + lam * adv_cost.mean() \
                     - ent_coef * ent

            opt_actor.zero_grad()
            a_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            opt_actor.step()

            v      = critic(obs).squeeze(-1)
            v_loss = F.mse_loss(v, ret_scalar)
            opt_critic.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            opt_critic.step()

            vc      = cost_critic(obs).squeeze(-1)
            vc_loss = F.mse_loss(vc, ret_cost)
            opt_cc.zero_grad()
            vc_loss.backward()
            torch.nn.utils.clip_grad_norm_(cost_critic.parameters(), max_grad_norm)
            opt_cc.step()

            metrics["loss_actor"]  += a_loss.item()
            metrics["loss_critic"] += v_loss.item()
            metrics["loss_cost_v"] += vc_loss.item()
            metrics["clip_frac"]   += ((ratio-1).abs() > clip_eps).float().mean().item()
            metrics["n"]           += 1.0

    assert metrics["n"] > 0, "No mini-batch updates; check n_epochs and batch_size."
    n = metrics.pop("n")
    return {k: v / n for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Baseline config
# ---------------------------------------------------------------------------

@dataclass
class BaselineConfig:
    dataset_path:    str   = "ai4i2020.xlsx"
    total_env_steps: int   = 2_000_000
    rollout_steps:   int   = 500
    n_epochs:        int   = 10
    batch_size:      int   = 64
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    lr:              float = 3e-4
    clip_eps:        float = 0.2
    ent_coef:        float = 0.01
    alpha_dual:      float = 3e-4
    max_grad_norm:   float = 0.5
    kappa: Dict[int, float] = field(
        default_factory=lambda: {0: 0.1040, 1: 0.0881, 2: 0.0847}
    )
    seed:   int   = 0
    device: str   = field(
        default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )
    log_every: int = 10


# ---------------------------------------------------------------------------
# Baseline 1 — PPO-Scalar
# ---------------------------------------------------------------------------

class PPOScalarTrainer:
    """
    Three independent per-cell PPO actors with a fixed scalarised reward.
    omega is fixed to [1/3, 1/3, 1/3].  No safety constraint.
    """

    NAME = "PPO-Scalar"

    def __init__(self, config: BaselineConfig):
        self.cfg    = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self._rng   = np.random.default_rng(config.seed)
        self.omega  = torch.full((N_OBJ,), 1/3, device=self.device)

        # Structural invariant: this class uses ONE scalar critic per cell.
        # The scalar value v is broadcast to shape (N_OBJ,) when stored in
        # RolloutBuffer so that compute_returns() produces the correct
        # scalarised advantage.  The math is valid because omega sums to 1:
        #   adv_scalar = (adv * omega).sum(-1)
        #              = sum_j omega_j * (r_j + gamma*v - v)
        #              = r_scalar + gamma*v - v   (standard TD error)
        # If N_OBJ is ever changed, verify that the torch.full((N_OBJ,), v)
        # calls in _collect() and set_last_values() below remain correct.
        assert N_OBJ >= 1, \
            f"PPOScalarTrainer: N_OBJ must be >= 1, got {N_OBJ}."

        self.env    = ManufacturingEnv(config.dataset_path, self._rng,
                                       max_steps=config.rollout_steps)
        self.actors   = [MLPActor().to(self.device) for _ in range(N_CELLS)]
        self.critics  = [MLPCritic().to(self.device) for _ in range(N_CELLS)]
        # No cost critics — no safety constraint
        self.opt_a    = [Adam(a.parameters(), lr=config.lr) for a in self.actors]
        self.opt_c    = [Adam(c.parameters(), lr=config.lr) for c in self.critics]
        self.buffers  = [
            RolloutBuffer(config.rollout_steps, config.gamma,
                          config.gae_lambda, self.device)
            for _ in range(N_CELLS)
        ]
        self.total_steps = 0

    @torch.no_grad()
    def _collect(self) -> None:
        for buf in self.buffers:
            buf.reset()
        obs_dict = self.env.reset(self.omega.cpu().numpy())
        obs_t    = {i: torch.as_tensor(obs_dict[i], dtype=torch.float32,
                                       device=self.device)
                    for i in range(N_CELLS)}

        for step in range(self.cfg.rollout_steps):
            actions_env  = {}
            # Collect per-cell data before env.step in a clean local dict.
            # raw_action is the unclipped Gaussian sample used for log_prob;
            # actions_env carries the clipped version sent to the environment.
            # Using the raw sample is consistent with trainer.py and ensures
            # that the stored log_prob matches the distribution the PPO ratio
            # is computed against during the update.
            per_cell: Dict[int, Dict] = {}
            for i in range(N_CELLS):
                obs_i       = obs_t[i].unsqueeze(0)
                mu_i, lsd_i = self.actors[i](obs_i)
                raw_i       = mu_i + lsd_i.exp() * torch.randn_like(mu_i)
                action_i    = torch.clamp(raw_i, -1.0, 1.0)
                lp_i        = _gaussian_log_prob(raw_i, mu_i, lsd_i).item()
                v_i         = self.critics[i](obs_i).item()
                actions_env[i] = action_i.squeeze(0).cpu().numpy()
                per_cell[i] = {
                    "raw_action": raw_i.squeeze(0),
                    "lp":         lp_i,
                    "v":          v_i,
                }

            next_obs, rewards, costs, done, info = self.env.step(actions_env)

            for i in range(N_CELLS):
                ci = info["cells"][i]
                rv = torch.tensor([ci["r1"], ci["r2"], ci["r3"]], dtype=torch.float32)
                # PPO-Scalar has one scalar critic per cell, not per-objective critics.
                # Broadcasting the scalar value v to shape (N_OBJ,) is mathematically
                # correct here: the multi-objective GAE in RolloutBuffer produces
                #   adv_scalar = (adv * omega).sum(-1)
                #              = sum_j omega_j * (r_j + gamma*v_next - v)
                #              = r_scalar + gamma*v_next - v
                # which is the standard TD error for the scalarised reward, exactly as
                # expected for a single-critic baseline.  The critic loss is also
                # computed on ret_scalar = (returns * omega).sum(-1), which reduces to
                # the standard lambda-return.  Individual per-objective returns are
                # not used elsewhere by this baseline.
                self.buffers[i].add(
                    obs        = obs_t[i],
                    action     = per_cell[i]["raw_action"],
                    log_prob   = per_cell[i]["lp"],
                    reward_vec = rv,
                    cost       = 0,
                    value_vec  = torch.full((N_OBJ,), per_cell[i]["v"]),
                    value_cost = 0.0,
                    done       = done,
                )
                obs_t[i] = torch.as_tensor(next_obs[i], dtype=torch.float32,
                                            device=self.device)
            # Do NOT reset the environment here.  obs_t now holds next_obs,
            # which is used as the bootstrap observation for set_last_values
            # below.  Resetting would overwrite obs_t with a fresh episode's
            # first state, making the bootstrap value inconsistent with the
            # job sequence that was actually run.  The done flag stored in the
            # buffer zeroes out the bootstrap in compute_returns(), so the
            # identity of the post-terminal state does not affect correctness
            # when done=True, but using next_obs is still more principled.

        for i in range(N_CELLS):
            lv = self.critics[i](obs_t[i].unsqueeze(0)).item()
            # Same broadcast rationale as above for the bootstrap value.
            self.buffers[i].set_last_values(torch.full((N_OBJ,), lv), 0.0)
            self.buffers[i].compute_returns(self.omega)
        self.total_steps += self.cfg.rollout_steps * N_CELLS

    def train(self, verbose: bool = True) -> None:
        while self.total_steps < self.cfg.total_env_steps:
            self._collect()
            for i in range(N_CELLS):
                _ppo_update_no_constraint(
                    actor=self.actors[i], critic=self.critics[i],
                    opt_actor=self.opt_a[i], opt_critic=self.opt_c[i],
                    buffer=self.buffers[i], omega=self.omega,
                    device=self.device, n_epochs=self.cfg.n_epochs,
                    batch_size=self.cfg.batch_size,
                    clip_eps=self.cfg.clip_eps, ent_coef=self.cfg.ent_coef,
                )
            if verbose and (self.total_steps // (self.cfg.rollout_steps * N_CELLS)) \
                    % self.cfg.log_every == 0:
                mean_costs = [self.buffers[i].mean_cost for i in range(N_CELLS)]
                print(f"  [{self.NAME}] steps={self.total_steps:,}  "
                      f"costs={[round(c,4) for c in mean_costs]}")

    @torch.no_grad()
    def evaluate(self, eval_steps: int = 500) -> pd.DataFrame:
        """Evaluate at all 45 simplex grid points."""
        grid  = build_simplex_grid(8)
        kappa = self.cfg.kappa
        rows  = []
        for omega_np in grid:
            omega_t = torch.as_tensor(omega_np, dtype=torch.float32,
                                      device=self.device)
            # Derive a unique seed per omega point from its integer representation
            # (same scheme as Evaluator._eval_one) so that each evaluation draws
            # an independent job sequence rather than all 45 sharing the same jobs.
            # No % 100_000: the modulo silently drops the o0 component and causes
            # seed collisions among grid points (see evaluator.py for details).
            omega_int = (int(round(omega_np[0] * 1000)) * 1_000_000
                         + int(round(omega_np[1] * 1000)) * 1_000
                         + int(round(omega_np[2] * 1000)))
            rng     = np.random.default_rng(self.cfg.seed + omega_int)
            env     = ManufacturingEnv(self.cfg.dataset_path, rng,
                                       max_steps=eval_steps)
            obs     = env.reset(omega_np)
            obs_t   = {i: torch.as_tensor(obs[i], dtype=torch.float32,
                                           device=self.device)
                       for i in range(N_CELLS)}
            ep_rvecs = {i: [] for i in range(N_CELLS)}
            ep_costs = {i: [] for i in range(N_CELLS)}
            done = False; step = 0
            while not done and step < eval_steps:
                actions_env = {}
                for i in range(N_CELLS):
                    mu_i, _ = self.actors[i](obs_t[i].unsqueeze(0))
                    actions_env[i] = torch.clamp(mu_i, -1, 1).squeeze(0).cpu().numpy()
                next_obs, _, costs, done, info = env.step(actions_env)
                for i in range(N_CELLS):
                    ci = info["cells"][i]
                    ep_rvecs[i].append([ci["r1"], ci["r2"], ci["r3"]])
                    ep_costs[i].append(float(costs[i]))
                    obs_t[i] = torch.as_tensor(next_obs[i], dtype=torch.float32,
                                               device=self.device)
                step += 1
            row = {"omega_0": float(omega_np[0]),
                   "omega_1": float(omega_np[1]),
                   "omega_2": float(omega_np[2])}
            all_rmeans = []
            for i in range(N_CELLS):
                rv = np.array(ep_rvecs[i])
                rm = rv.mean(axis=0)
                row[f"cell{i}_r1"] = float(rm[0])
                row[f"cell{i}_r2"] = float(rm[1])
                row[f"cell{i}_r3"] = float(rm[2])
                row[f"cell{i}_cost"] = float(np.mean(ep_costs[i]))
                row[f"cell{i}_feasible"] = float(
                    np.mean(ep_costs[i]) < kappa[i])
                all_rmeans.append(rm)
            mean_obj = np.mean(all_rmeans, axis=0)
            row["mean_r1"]     = float(mean_obj[0])
            row["mean_r2"]     = float(mean_obj[1])
            row["mean_r3"]     = float(mean_obj[2])
            row["utility"]     = float(omega_np @ mean_obj)
            row["all_feasible"] = float(all(
                np.mean(ep_costs[i]) < kappa[i] for i in range(N_CELLS)))
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baseline 2 — MACPO (MARL + Lagrangian, fixed omega)
# ---------------------------------------------------------------------------

class MACPOTrainer(PPOScalarTrainer):
    """
    Multi-Agent CPO approximation: same as PPO-Scalar but WITH per-cell
    Lagrangian safety constraints.
    """

    NAME = "MACPO"

    def __init__(self, config: BaselineConfig):
        super().__init__(config)
        self.cost_critics = [MLPCritic().to(self.device) for _ in range(N_CELLS)]
        self.opt_cc       = [Adam(cc.parameters(), lr=config.lr)
                             for cc in self.cost_critics]
        self.lag_updater  = LagrangianUpdater(config.kappa, config.alpha_dual)

    def train(self, verbose: bool = True) -> None:
        while self.total_steps < self.cfg.total_env_steps:
            self._collect_with_cost()
            lambdas = self.lag_updater.lambdas()
            for i in range(N_CELLS):
                _ppo_update_with_constraint(
                    actor=self.actors[i], critic=self.critics[i],
                    cost_critic=self.cost_critics[i],
                    opt_actor=self.opt_a[i], opt_critic=self.opt_c[i],
                    opt_cc=self.opt_cc[i],
                    buffer=self.buffers[i], omega=self.omega,
                    lam=lambdas[i], device=self.device,
                    n_epochs=self.cfg.n_epochs, batch_size=self.cfg.batch_size,
                    clip_eps=self.cfg.clip_eps, ent_coef=self.cfg.ent_coef,
                )
            mean_costs = {i: self.buffers[i].mean_cost for i in range(N_CELLS)}
            self.lag_updater.update(mean_costs)
            if verbose and (self.total_steps // (self.cfg.rollout_steps * N_CELLS)) \
                    % self.cfg.log_every == 0:
                lams = self.lag_updater.lambdas()
                print(f"  [{self.NAME}] steps={self.total_steps:,}  "
                      f"costs={[round(mean_costs[i],4) for i in range(N_CELLS)]}  "
                      f"lambdas={[round(lams[i],5) for i in range(N_CELLS)]}")

    @torch.no_grad()
    def _collect_with_cost(self) -> None:
        """Collect rollout storing real cost signals."""
        for buf in self.buffers:
            buf.reset()
        obs_dict = self.env.reset(self.omega.cpu().numpy())
        obs_t    = {i: torch.as_tensor(obs_dict[i], dtype=torch.float32,
                                       device=self.device)
                    for i in range(N_CELLS)}
        for step in range(self.cfg.rollout_steps):
            actions_env = {}
            per_cell_data = {}
            for i in range(N_CELLS):
                obs_i       = obs_t[i].unsqueeze(0)
                mu_i, lsd_i = self.actors[i](obs_i)
                raw_i       = mu_i + lsd_i.exp() * torch.randn_like(mu_i)
                act_i       = torch.clamp(raw_i, -1.0, 1.0)
                lp_i        = _gaussian_log_prob(raw_i, mu_i, lsd_i).item()
                v_i         = self.critics[i](obs_i).item()
                vc_i        = self.cost_critics[i](obs_i).item()
                actions_env[i] = act_i.squeeze(0).cpu().numpy()
                # Store the raw (unclipped) sample so the importance ratio in
                # the PPO update is computed against the correct Gaussian.
                per_cell_data[i] = (raw_i.squeeze(0), lp_i, v_i, vc_i)

            next_obs, _, costs, done, info = self.env.step(actions_env)

            for i in range(N_CELLS):
                raw_act_i, lp_i, v_i, vc_i = per_cell_data[i]
                ci  = info["cells"][i]
                rv  = torch.tensor([ci["r1"], ci["r2"], ci["r3"]],
                                   dtype=torch.float32)
                # MACPO, like PPO-Scalar, uses a single scalar critic per cell.
                # Broadcasting v_i to (N_OBJ,) is correct: the downstream GAE
                # produces the right scalarised advantage and lambda-return
                # (see the detailed comment in PPOScalarTrainer._collect).
                self.buffers[i].add(
                    obs=obs_t[i], action=raw_act_i, log_prob=lp_i,
                    reward_vec=rv, cost=costs[i],
                    value_vec=torch.full((N_OBJ,), v_i),
                    value_cost=vc_i, done=done,
                )
                obs_t[i] = torch.as_tensor(next_obs[i], dtype=torch.float32,
                                            device=self.device)
            # Do NOT reset here — same rationale as PPOScalarTrainer._collect().

        for i in range(N_CELLS):
            lv  = self.critics[i](obs_t[i].unsqueeze(0)).item()
            lvc = self.cost_critics[i](obs_t[i].unsqueeze(0)).item()
            # Same broadcast rationale as above for the bootstrap value.
            self.buffers[i].set_last_values(torch.full((N_OBJ,), lv), lvc)
            self.buffers[i].compute_returns(self.omega)
        self.total_steps += self.cfg.rollout_steps * N_CELLS


# ---------------------------------------------------------------------------
# Baseline 4 — CAPQL (centralised single-agent safe MORL)
# ---------------------------------------------------------------------------

class CAPQLTrainer:
    """
    Centralised actor with joint observations + omega as input.
    Single Lagrange multiplier shared across all cells.
    Omega is fixed per training run.
    """

    NAME = "CAPQL"

    def __init__(self, config: BaselineConfig):
        self.cfg    = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        self._rng   = np.random.default_rng(config.seed)
        self.omega  = torch.full((N_OBJ,), 1/3, device=self.device)

        self.env = ManufacturingEnv(config.dataset_path, self._rng,
                                    max_steps=config.rollout_steps)

        # Input: joint obs (N_CELLS * OBS_DIM) + omega (N_OBJ)
        # Output: one action delta per cell (N_CELLS), not a single scalar.
        self._joint_input_dim = N_CELLS * OBS_DIM + N_OBJ
        self.actor       = MLPActor(self._joint_input_dim,
                                    output_dim=N_CELLS).to(self.device)
        self.critic      = MLPCritic(self._joint_input_dim).to(self.device)
        self.cost_critic = MLPCritic(self._joint_input_dim).to(self.device)

        self.opt_a   = Adam(self.actor.parameters(),       lr=config.lr)
        self.opt_c   = Adam(self.critic.parameters(),      lr=config.lr)
        self.opt_cc  = Adam(self.cost_critic.parameters(), lr=config.lr)

        # Single shared Lagrange multiplier
        # (CAPQL uses one multiplier for the joint constraint violation)
        self._lambda = 0.0

        # Collect joint transitions in a single buffer (one per step, not per cell).
        # We repurpose RolloutBuffer and override two tensor shapes so that:
        #   obs     stores the joint observation vector (joint_input_dim per step)
        #   actions stores N_CELLS action values per step (one per cell)
        self.buffer = RolloutBuffer(
            config.rollout_steps, config.gamma, config.gae_lambda, self.device
        )
        T = config.rollout_steps
        self.buffer.obs     = torch.zeros(T, self._joint_input_dim, device=self.device)
        self.buffer.actions = torch.zeros(T, N_CELLS,               device=self.device)
        self.total_steps = 0

    def _make_joint_obs(self, obs_dict: Dict, omega: Tensor) -> Tensor:
        """Concatenate all cell obs and omega into one vector."""
        parts = [torch.as_tensor(obs_dict[i], dtype=torch.float32,
                                 device=self.device)
                 for i in range(N_CELLS)]
        parts.append(omega)
        return torch.cat(parts)   # (N_CELLS*OBS_DIM + N_OBJ,)

    @torch.no_grad()
    def _collect(self) -> None:
        self.buffer.reset()
        # RolloutBuffer.reset() only resets the write-pointer and flags;
        # it does NOT touch the tensor shapes.  The custom shapes set in
        # __init__ (joint_input_dim for obs, N_CELLS for actions) survive
        # reset() unchanged, so no re-allocation is needed here.
        T = self.cfg.rollout_steps

        obs_dict    = self.env.reset(self.omega.cpu().numpy())
        joint_obs_t = self._make_joint_obs(obs_dict, self.omega)

        for step in range(T):
            obs_i   = joint_obs_t.unsqueeze(0)
            mu, lsd = self.actor(obs_i)           # mu: (1, N_CELLS)

            # Unclipped sample for correct log_prob; clipped for the environment.
            raw = mu + lsd.exp() * torch.randn_like(mu)   # (1, N_CELLS)
            act = torch.clamp(raw, -1.0, 1.0)             # (1, N_CELLS)

            # Joint log-probability over all N_CELLS actions (sum over dim=-1).
            lp  = _gaussian_log_prob(raw, mu, lsd).item()
            v   = self.critic(obs_i).item()
            vc  = self.cost_critic(obs_i).item()

            # Send a distinct action to each cell from the N_CELLS output vector.
            act_np      = act.squeeze(0).cpu().numpy()    # (N_CELLS,)
            actions_env = {i: act_np[i:i+1] for i in range(N_CELLS)}
            next_obs, _, costs, done, info = self.env.step(actions_env)

            # Mean reward across cells, mean cost across cells
            r_vecs    = [[info["cells"][i]["r1"],
                          info["cells"][i]["r2"],
                          info["cells"][i]["r3"]] for i in range(N_CELLS)]
            rv_mean   = torch.tensor(np.mean(r_vecs, axis=0), dtype=torch.float32)
            cost_mean = float(np.mean([costs[i] for i in range(N_CELLS)]))

            # CAPQL uses non-standard tensor shapes, so we bypass buffer.add()
            # and write fields directly.  The raw (unclipped) action is stored
            # so the PPO importance ratio re-evaluates the correct distribution.
            # value_vec is broadcast from the scalar critic (same rationale as
            # PPOScalarTrainer._collect; see the comment there for the math).
            ptr = self.buffer._ptr
            self.buffer.obs[ptr]        = joint_obs_t.to(self.buffer.device).float()
            self.buffer.actions[ptr]    = raw.squeeze(0).to(self.buffer.device).float()
            self.buffer.log_probs[ptr]  = float(lp)
            self.buffer.reward_vec[ptr] = rv_mean.to(self.buffer.device).float()
            self.buffer.costs[ptr]      = float(cost_mean)
            self.buffer.value_vec[ptr]  = torch.full((N_OBJ,), v,
                                              device=self.buffer.device)
            self.buffer.value_cost[ptr] = float(vc)
            self.buffer.dones[ptr]      = float(done)
            self.buffer._ptr += 1
            if self.buffer._ptr >= self.buffer.capacity:
                self.buffer._full = True

            joint_obs_t = self._make_joint_obs(
                {i: next_obs[i] for i in range(N_CELLS)}, self.omega
            )
            # Do NOT reset here — same rationale as PPOScalarTrainer._collect().

        lv  = self.critic(joint_obs_t.unsqueeze(0)).item()
        lvc = self.cost_critic(joint_obs_t.unsqueeze(0)).item()
        # Same broadcast rationale for the bootstrap value.
        self.buffer.set_last_values(torch.full((N_OBJ,), lv), lvc)
        self.buffer.compute_returns(self.omega)
        self.total_steps += self.cfg.rollout_steps * N_CELLS

    def train(self, verbose: bool = True) -> None:
        while self.total_steps < self.cfg.total_env_steps:
            self._collect()
            _ppo_update_with_constraint(
                actor=self.actor, critic=self.critic,
                cost_critic=self.cost_critic,
                opt_actor=self.opt_a, opt_critic=self.opt_c,
                opt_cc=self.opt_cc,
                buffer=self.buffer, omega=self.omega,
                lam=self._lambda, device=self.device,
                n_epochs=self.cfg.n_epochs,
                batch_size=self.cfg.batch_size,
                clip_eps=self.cfg.clip_eps, ent_coef=self.cfg.ent_coef,
            )
            # Dual update on shared multiplier
            cost_mean = self.buffer.mean_cost
            # Mean kappa across cells
            kappa_mean = np.mean(list(self.cfg.kappa.values()))
            self._lambda = max(0.0, self._lambda +
                               self.cfg.alpha_dual * (cost_mean - kappa_mean))
            if verbose and (self.total_steps // (self.cfg.rollout_steps * N_CELLS)) \
                    % self.cfg.log_every == 0:
                print(f"  [{self.NAME}] steps={self.total_steps:,}  "
                      f"cost={cost_mean:.4f}  lambda={self._lambda:.5f}")

    @torch.no_grad()
    def evaluate(self, eval_steps: int = 500) -> pd.DataFrame:
        """Evaluate at all 45 simplex grid points (omega varies at eval only)."""
        grid  = build_simplex_grid(8)
        kappa = self.cfg.kappa
        rows  = []
        for omega_np in grid:
            omega_t     = torch.as_tensor(omega_np, dtype=torch.float32,
                                          device=self.device)
            # Derive a unique seed per omega point from its integer representation
            # so each evaluation draws an independent job sequence.
            # No % 100_000: the modulo causes seed collisions (see evaluator.py).
            omega_int   = (int(round(omega_np[0] * 1000)) * 1_000_000
                           + int(round(omega_np[1] * 1000)) * 1_000
                           + int(round(omega_np[2] * 1000)))
            rng         = np.random.default_rng(self.cfg.seed + omega_int)
            env         = ManufacturingEnv(self.cfg.dataset_path, rng,
                                           max_steps=eval_steps)
            obs_dict    = env.reset(omega_np)
            joint_obs_t = self._make_joint_obs(obs_dict, omega_t)

            ep_rvecs    = {i: [] for i in range(N_CELLS)}
            ep_costs    = {i: [] for i in range(N_CELLS)}
            done = False; step = 0
            while not done and step < eval_steps:
                mu_a, _ = self.actor(joint_obs_t.unsqueeze(0))
                act_a   = torch.clamp(mu_a, -1.0, 1.0)   # (1, N_CELLS)
                act_np  = act_a.squeeze(0).cpu().numpy()  # (N_CELLS,)
                # Send a distinct mean action to each cell (fix: not a broadcast).
                actions_env = {i: act_np[i:i+1] for i in range(N_CELLS)}
                next_obs, _, costs, done, info = env.step(actions_env)
                for i in range(N_CELLS):
                    ci = info["cells"][i]
                    ep_rvecs[i].append([ci["r1"], ci["r2"], ci["r3"]])
                    ep_costs[i].append(float(costs[i]))
                joint_obs_t = self._make_joint_obs(
                    {i: next_obs[i] for i in range(N_CELLS)}, omega_t
                )
                step += 1

            row = {"omega_0": float(omega_np[0]),
                   "omega_1": float(omega_np[1]),
                   "omega_2": float(omega_np[2])}
            all_rmeans = []
            for i in range(N_CELLS):
                rv = np.array(ep_rvecs[i]); rm = rv.mean(axis=0)
                row[f"cell{i}_r1"] = float(rm[0])
                row[f"cell{i}_r2"] = float(rm[1])
                row[f"cell{i}_r3"] = float(rm[2])
                row[f"cell{i}_cost"]     = float(np.mean(ep_costs[i]))
                row[f"cell{i}_feasible"] = float(np.mean(ep_costs[i]) < kappa[i])
                all_rmeans.append(rm)
            mean_obj = np.mean(all_rmeans, axis=0)
            row["mean_r1"]      = float(mean_obj[0])
            row["mean_r2"]      = float(mean_obj[1])
            row["mean_r3"]      = float(mean_obj[2])
            row["utility"]      = float(omega_np @ mean_obj)
            row["all_feasible"] = float(all(
                np.mean(ep_costs[i]) < kappa[i] for i in range(N_CELLS)))
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Convenience: build all four baselines
# ---------------------------------------------------------------------------

def build_baselines(config: BaselineConfig) -> Dict:
    return {
        "PPO-Scalar": PPOScalarTrainer(config),
        "MACPO":      MACPOTrainer(config),
        "CAPQL":      CAPQLTrainer(config),
    }


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== baselines.py  smoke test (minimal steps) ===\n")

    cfg = BaselineConfig(
        dataset_path    = "ai4i2020.xlsx",
        total_env_steps = 500 * N_CELLS * 2,   # 2 rollouts each
        rollout_steps   = 500,
        n_epochs        = 2,
        batch_size      = 64,
        seed            = 0,
        log_every       = 1,
    )   # device auto-detected via BaselineConfig default_factory

    for name, trainer_cls in [("PPO-Scalar", PPOScalarTrainer),
                               ("MACPO",      MACPOTrainer),
                               ("CAPQL",      CAPQLTrainer)]:
        print(f"--- {name} ---")
        t = trainer_cls(cfg)
        t.train(verbose=True)

        # Evaluate 3 grid points only for speed
        grid = build_simplex_grid(8)
        if name == "CAPQL":
            rows = []
            for k in [0, 22, 44]:
                omega_np = grid[k]
                omega_t  = torch.as_tensor(omega_np, dtype=torch.float32)
                rng      = np.random.default_rng(0)
                env      = ManufacturingEnv(cfg.dataset_path, rng, max_steps=50)
                obs_dict = env.reset(omega_np)
                joint_obs = t._make_joint_obs(obs_dict, omega_t)
                ep_rv = {i: [] for i in range(N_CELLS)}
                done = False; step = 0
                while not done and step < 50:
                    mu_a, _ = t.actor(joint_obs.unsqueeze(0))
                    act_a   = torch.clamp(mu_a, -1, 1)
                    act_np  = act_a.squeeze(0).cpu().numpy()          # (N_CELLS,)
                    ae      = {i: act_np[i:i+1] for i in range(N_CELLS)}  # per-cell slice
                    no, _, _, done, info = env.step(ae)
                    for i in range(N_CELLS):
                        ci = info["cells"][i]
                        ep_rv[i].append([ci["r1"], ci["r2"], ci["r3"]])
                    joint_obs = t._make_joint_obs(
                        {i: no[i] for i in range(N_CELLS)}, omega_t)
                    step += 1
                mean_obj = np.mean([np.array(ep_rv[i]).mean(axis=0)
                                    for i in range(N_CELLS)], axis=0)
                rows.append({"mean_r1": mean_obj[0], "mean_r2": mean_obj[1],
                             "mean_r3": mean_obj[2], "utility": omega_np @ mean_obj,
                             "omega_0": omega_np[0], "omega_1": omega_np[1],
                             "omega_2": omega_np[2],
                             **{f"cell{i}_cost": 0.1 for i in range(N_CELLS)},
                             **{f"cell{i}_feasible": 0.0 for i in range(N_CELLS)},
                             "all_feasible": 0.0})
            df = pd.DataFrame(rows)
        else:
            df_full = t.evaluate(eval_steps=50)
            df = pd.DataFrame([df_full.iloc[k].to_dict() for k in [0, 22, 44]])

        from safepc_marl.evaluator import Evaluator as _Ev, _hypervolume_of_set
        import numpy as _np
        obj = df[["mean_r1", "mean_r2", "mean_r3"]].values
        hv  = _hypervolume_of_set(obj)
        print(f"  HV (3 pts, untrained): {hv:.6f}")
        print(f"  mean_r1: {df['mean_r1'].values.round(3)}")
        print()

    print("Smoke test PASSED.")
