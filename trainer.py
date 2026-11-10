"""
trainer.py
==========
SafePC-MARL trainer.

Orchestrates the full training loop:

    collect_rollout()  ->  PPO update  ->  Lagrangian update  ->  log  ->  repeat

Public entry point
------------------
    trainer = Trainer(config)
    trainer.train()

All hyperparameters are collected in TrainerConfig, a frozen dataclass.
This keeps the constructor clean and makes config serialisation trivial.

Rollout collection details
--------------------------
At each step the trainer acts under the CURRENT policy (no gradient):
  1. Call hnet(omega) to get per-cell thetas and log_stds.
  2. For each cell, call PolicyNet.forward(obs, theta) to get mu.
  3. Sample action ~ N(mu, exp(log_std)).  Clip to [-1, 1].
  4. Compute log_prob.
  5. Call critic(joint_obs, omega) to get value_vec per cell.
  6. Call cost_critics[i](obs) to get value_cost per cell.
  7. Store all fields in the per-cell RolloutBuffer.
  8. After T steps, bootstrap last values and call compute_returns().

Joint critic update
-------------------
After each PPO epoch, update_critic_joint() is called with the full
T-step sequence of joint observations and the corresponding λ-return
targets (averaged over objectives and cells).  This is the only place
where the centralised critic sees all cells' observations simultaneously,
consistent with CTDE.

Omega sampling strategy
-----------------------
During training, omega is resampled from Dirichlet(1,1,1) at the start
of each rollout.  This ensures the HyperNetwork sees the full simplex
during training, not just a fixed omega.  The paper's 45 evaluation
points on the simplex are used only during evaluation.

Logging
-------
A simple CSV log is written after each update.  Every logged scalar is
the mean over the rollout.  No external dependency (no wandb/tensorboard)
is required; the CSV can be read with pandas for analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import math
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from safepc_marl.networks import (
    HyperNetwork, CentralizedCritic, CostCritic, PolicyNet,
    build_networks,
    N_CELLS, N_OBJ, OBS_DIM, OMEGA_DIM,
)
from safepc_marl.buffer   import RolloutBuffer
from safepc_marl.updater  import PPOUpdater, LagrangianUpdater, _gaussian_log_prob
from safepc_marl.simulator import ManufacturingEnv, VARIANTS


# ---------------------------------------------------------------------------
# Trainer configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainerConfig:
    """All hyperparameters for one training run."""

    # Environment
    dataset_path:   str   = "ai4i2020.xlsx"
    max_steps_ep:   int   = 500        # steps per episode (= rollout length T)
    seed:           int   = 0

    # Training schedule
    # NOTE: run.py's --steps CLI argument also defaults to 4_000_000 and always
    # overrides this value when using the standard entry point.  The dataclass
    # default is kept in sync so that creating TrainerConfig() directly (e.g.
    # in unit tests) uses the same full-run step count as the CLI.
    total_env_steps: int  = 4_000_000  # total steps across all cells
    rollout_steps:   int  = 500        # steps collected per rollout (= T)

    # PPO
    n_epochs:       int   = 10
    batch_size:     int   = 64
    gamma:          float = 0.99
    gae_lambda:     float = 0.95
    clip_eps:       float = 0.2
    vf_coef:        float = 0.5
    ent_coef:       float = 0.01
    alpha_div:      float = 0.05
    max_grad_norm:  float = 0.5

    # Learning rates
    lr_actor:       float = 3e-4
    lr_critic:      float = 1e-3
    lr_cost_critic: float = 1e-3

    # Lagrangian
    alpha_dual:     float = 3e-4
    lambda_init:    float = 0.0

    # Safety budgets (empirical, from dataset.py)
    kappa: Dict[int, float] = field(
        default_factory=lambda: {0: 0.1040, 1: 0.0881, 2: 0.0847}
    )

    # Omega sampling: fixed omega for the whole run, or resample per rollout?
    # "dirichlet" = resample from Dir(1,1,1) each rollout (recommended for training)
    # "fixed"     = use omega_fixed throughout
    omega_mode:     str   = "dirichlet"
    omega_fixed:    List[float] = field(default_factory=lambda: [1/3, 1/3, 1/3])

    # Logging and checkpointing
    log_dir:        str   = "runs"
    log_every:      int   = 10         # log every N updates
    save_every:     int   = 50         # checkpoint every N updates
    run_name:       str   = "safepc_marl"
    # device: auto-detect at construction time so that creating TrainerConfig()
    # directly (without going through run.py) works on CPU-only machines.
    # run.py also does its own detection and passes the result here, which
    # overrides this default.
    device:         str   = field(
        default_factory=lambda: "cuda" if __import__("torch").cuda.is_available() else "cpu"
    )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Manages the full training loop for SafePC-MARL.

    Parameters
    ----------
    config : TrainerConfig
    """

    def __init__(self, config: TrainerConfig):
        self.cfg = config
        self.device = torch.device(config.device)

        # --- Reproducibility ---
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self._rng = np.random.default_rng(config.seed)

        # --- Environment ---
        self.env = ManufacturingEnv(
            dataset_path = config.dataset_path,
            rng          = self._rng,
            max_steps    = config.max_steps_ep,
        )

        # --- Networks ---
        nets = build_networks(self.device)
        self.hnet        = nets["hnet"]
        self.critic      = nets["critic"]
        self.cost_critics: List[CostCritic] = nets["cost_critics"]

        # --- Lagrangian ---
        self.lag_updater = LagrangianUpdater(
            kappa       = config.kappa,
            alpha_dual  = config.alpha_dual,
            lambda_init = config.lambda_init,
        )

        # --- PPO updater ---
        self.updater = PPOUpdater(
            hnet           = self.hnet,
            critic         = self.critic,
            cost_critics   = self.cost_critics,
            lag_updater    = self.lag_updater,
            lr_actor       = config.lr_actor,
            lr_critic      = config.lr_critic,
            lr_cost_critic = config.lr_cost_critic,
            clip_eps       = config.clip_eps,
            n_epochs       = config.n_epochs,
            batch_size     = config.batch_size,
            vf_coef        = config.vf_coef,
            ent_coef       = config.ent_coef,
            alpha_div      = config.alpha_div,
            max_grad_norm  = config.max_grad_norm,
            device         = self.device,
        )

        # --- Rollout buffers (one per cell) ---
        self.buffers: List[RolloutBuffer] = [
            RolloutBuffer(
                capacity   = config.rollout_steps,
                gamma      = config.gamma,
                gae_lambda = config.gae_lambda,
                device     = self.device,
            )
            for _ in range(N_CELLS)
        ]

        # --- Logging ---
        log_dir = Path(config.log_dir) / config.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path  = log_dir / "training_log.csv"
        self.ckpt_dir  = log_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True)
        # CSV file is opened eagerly so the log file is created immediately
        # (makes the run directory visible from the first moment of the run).
        # _close_log() is called at the end of train() and in __del__ as a
        # safety net for --eval-only mode or mid-training exceptions.
        self._csv_file   = open(self.log_path, "w", newline="")
        self._csv_writer: csv.DictWriter | None = None

        # --- State ---
        self.total_steps  = 0   # environment steps taken across all cells
        self.n_updates    = 0   # number of PPO update calls
        self._omega       = self._sample_omega()

    def __del__(self) -> None:
        """
        Safety net: close the CSV log file if it is still open.

        This handles two scenarios where train() might never be called or
        might not finish:
          1. ``--eval-only`` mode: only Trainer.__init__ and load() run.
          2. An exception raised mid-training before _close_log() is reached.

        Python's garbage collector calls __del__ when the Trainer is
        deallocated, guaranteeing the file descriptor is released even if the
        normal teardown path was skipped.
        """
        try:
            if hasattr(self, "_csv_file") and not self._csv_file.closed:
                self._csv_file.close()
        except Exception:
            pass   # never raise from __del__

    # ------------------------------------------------------------------
    # Omega management
    # ------------------------------------------------------------------

    def _sample_omega(self) -> Tensor:
        """
        Sample a preference vector for the upcoming rollout.

        Strategy (omega_mode == 'dirichlet'):
        ──────────────────────────────────────
        • 80 % of rollouts: Dirichlet(1, 1, 1)  — uniform on the simplex.
            Provides balanced coverage of all operator preference settings.

        • 20 % of rollouts: corner-concentrated Dirichlet(5, 0.1, 0.1) or
            a permutation thereof.  Forces the HyperNetwork to specialise
            at the three simplex corners — the hardest points for the oracle
            comparison in the switching-degradation metric.
            Net corner coverage (any omega_i > 0.7): ~38 %.
        """
        if self.cfg.omega_mode == "fixed":
            w = torch.tensor(self.cfg.omega_fixed, dtype=torch.float32)
        else:
            p = self._rng.random()
            if p < 0.20:
                # Corner-concentrated sample
                corner   = int(self._rng.integers(0, OMEGA_DIM))
                alpha_np = np.full(OMEGA_DIM, 0.1)
                alpha_np[corner] = 5.0
                alpha_t  = torch.tensor(alpha_np, dtype=torch.float32)
                w        = torch.distributions.Dirichlet(alpha_t).sample()
            else:
                # Uniform on the simplex
                alpha = torch.ones(OMEGA_DIM)
                w     = torch.distributions.Dirichlet(alpha).sample()
        return w.to(self.device)

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect_rollout(self, omega: Tensor) -> Dict:
        """
        Collect exactly `rollout_steps` environment steps for all cells.

        The policy and critics are queried with no gradient.  All results
        are stored in self.buffers.

        Parameters
        ----------
        omega : (N_OBJ,)  — preference vector for this rollout

        Returns
        -------
        rollout_info : dict with mean reward, cost, etc. for logging
        """
        # Reset buffers
        for buf in self.buffers:
            buf.reset()

        # Reset environment
        obs_dict = self.env.reset(omega.cpu().numpy())

        # Convert initial observations to tensors
        obs_tensors: Dict[int, Tensor] = {
            i: torch.as_tensor(obs_dict[i], dtype=torch.float32,
                               device=self.device)
            for i in range(N_CELLS)
        }

        # Accumulators for rollout-level statistics
        ep_rewards   = {i: [] for i in range(N_CELLS)}
        ep_costs     = {i: [] for i in range(N_CELLS)}
        ep_r_vecs    = {i: [] for i in range(N_CELLS)}

        T = self.cfg.rollout_steps

        for step in range(T):
            # ---------------------------------------------------------
            # 1. Get policy parameters from HyperNetwork
            # ---------------------------------------------------------
            thetas, log_stds = self.hnet(omega)

            # ---------------------------------------------------------
            # 2. Get value estimates for all cells (joint observation)
            # ---------------------------------------------------------
            joint_obs = torch.cat(
                [obs_tensors[i] for i in range(N_CELLS)], dim=0
            ).unsqueeze(0)   # (1, N_CELLS * OBS_DIM)

            values_all = self.critic(joint_obs, omega)   # (1, N_CELLS, N_OBJ)
            values_all = values_all.squeeze(0)           # (N_CELLS, N_OBJ)

            # ---------------------------------------------------------
            # 3. Act in environment
            # ---------------------------------------------------------
            actions_env: Dict[int, np.ndarray] = {}
            action_tensors: Dict[int, Tensor]  = {}
            log_probs_step: Dict[int, float]   = {}
            value_costs_step: Dict[int, float] = {}

            for i in range(N_CELLS):
                obs_i    = obs_tensors[i].unsqueeze(0)   # (1, OBS_DIM)
                theta_i  = thetas[i]                     # (POLICY_PARAM_DIM,)
                log_std_i = log_stds[i]                  # scalar

                # Sample action from Gaussian policy
                mu_i      = PolicyNet.forward(obs_i, theta_i)   # (1, 1)
                std_i     = log_std_i.exp()
                eps       = torch.randn_like(mu_i)
                raw_action = mu_i + std_i * eps                  # unclipped sample
                action_i   = torch.clamp(raw_action, -1.0, 1.0) # clipped for env

                # Log probability under the current policy.
                # IMPORTANT: compute log_prob on the RAW (unclipped) sample so it
                # matches the Gaussian distribution.  The clipped value is only used
                # as the environment input; the importance ratio uses the raw sample.
                lp_i = _gaussian_log_prob(raw_action, mu_i, log_std_i)  # (1,)

                # Cost critic
                vc_i = self.cost_critics[i](obs_i).item()

                action_tensors[i]    = raw_action.squeeze(0)  # (1,) — raw for buffer/ratio
                log_probs_step[i]    = lp_i.item()
                value_costs_step[i]  = vc_i
                actions_env[i]       = action_i.squeeze(0).cpu().numpy()  # clipped for env

            # Step environment
            next_obs_dict, rewards, costs, done, info = self.env.step(actions_env)

            # ---------------------------------------------------------
            # 4. Store transitions
            # ---------------------------------------------------------
            for i in range(N_CELLS):
                cell_info  = info["cells"][i]
                reward_vec = torch.tensor(
                    [cell_info["r1"], cell_info["r2"], cell_info["r3"]],
                    dtype=torch.float32,
                )

                self.buffers[i].add(
                    obs        = obs_tensors[i],
                    action     = action_tensors[i],
                    log_prob   = log_probs_step[i],
                    reward_vec = reward_vec,
                    cost       = costs[i],
                    value_vec  = values_all[i],
                    value_cost = value_costs_step[i],
                    done       = done,
                )

                ep_rewards[i].append(rewards[i])
                ep_costs[i].append(float(costs[i]))
                ep_r_vecs[i].append(reward_vec.numpy())

            # Advance observations to the state returned by env.step().
            # When done=True the environment already advanced its internal job
            # sampler, so next_obs_dict[i] holds the first observation of the
            # next simulated job — this is the correct state from which to
            # bootstrap V(s_{T}) in the GAE computation.  The done flag stored
            # in the buffer (done=True for step T-1) will zero out that
            # bootstrap value in compute_returns(), so it does not matter that
            # it comes from a "new episode" state rather than a terminal state.
            # Importantly, we must NOT reset the environment here: doing so
            # would overwrite obs_tensors with the first state of a freshly
            # reset episode (a different RNG draw than the simulator's internal
            # next job), making the bootstrap value computed below inconsistent
            # with the job sequence that was actually run.
            for i in range(N_CELLS):
                obs_tensors[i] = torch.as_tensor(
                    next_obs_dict[i], dtype=torch.float32, device=self.device
                )

        # -----------------------------------------------------------------
        # 5. Bootstrap last values
        # -----------------------------------------------------------------
        joint_obs_last = torch.cat(
            [obs_tensors[i] for i in range(N_CELLS)], dim=0
        ).unsqueeze(0)

        last_values_all = self.critic(joint_obs_last, omega).squeeze(0)  # (N_CELLS, N_OBJ)

        for i in range(N_CELLS):
            last_vc = self.cost_critics[i](obs_tensors[i].unsqueeze(0)).item()
            self.buffers[i].set_last_values(last_values_all[i], last_vc)
            self.buffers[i].compute_returns(omega)

        # -----------------------------------------------------------------
        # 6. Summary statistics
        # -----------------------------------------------------------------
        self.total_steps += T * N_CELLS

        info_out: Dict = {
            "omega": omega.cpu().tolist(),
        }
        for i in range(N_CELLS):
            v = VARIANTS[i]
            r_arr = np.array(ep_r_vecs[i])  # (T, N_OBJ)
            info_out[f"cell{i}_mean_reward"]    = float(np.mean(ep_rewards[i]))
            info_out[f"cell{i}_mean_cost"]      = float(np.mean(ep_costs[i]))
            info_out[f"cell{i}_mean_r1"]        = float(r_arr[:, 0].mean())
            info_out[f"cell{i}_mean_r2"]        = float(r_arr[:, 1].mean())
            info_out[f"cell{i}_mean_r3"]        = float(r_arr[:, 2].mean())
            info_out[f"cell{i}_constraint_viol"]= float(
                np.mean(ep_costs[i]) - self.cfg.kappa[i]
            )

        return info_out

    # ------------------------------------------------------------------
    # Joint critic update (full CTDE pass)
    # ------------------------------------------------------------------

    def _joint_critic_update(self, omega: Tensor, batch_size: int | None = None) -> float:
        """
        Update the centralised critic using joint observations from all three
        cell buffers, aligned by time step.

        Parameters
        ----------
        omega      : (N_OBJ,)
        batch_size : mini-batch size. Defaults to self.cfg.batch_size so the
                     critic uses the SAME batch size as the actor update.

        Returns
        -------
        mean_loss : float
        """
        if batch_size is None:
            batch_size = self.cfg.batch_size
        T = self.buffers[0].size

        # Build joint obs and target tensors
        joint_obs = torch.cat(
            [self.buffers[i].obs[:T] for i in range(N_CELLS)], dim=-1
        )   # (T, N_CELLS * OBS_DIM)

        returns_all = torch.stack(
            [self.buffers[i].returns[:T] for i in range(N_CELLS)], dim=1
        )   # (T, N_CELLS, N_OBJ)

        # Mini-batch update
        total_loss = 0.0
        n_batches  = 0
        perm       = torch.randperm(T, device=self.device)

        for start in range(0, T, batch_size):
            idx = perm[start : start + batch_size]
            loss = self.updater.update_critic_joint(
                joint_obs  = joint_obs[idx].to(self.device),
                returns    = returns_all[idx].to(self.device),
                omega      = omega,
            )
            total_loss += loss
            n_batches  += 1

        assert n_batches > 0, \
            f"_joint_critic_update produced zero batches (T={T}, batch_size={batch_size})."
        return total_loss / n_batches

    # ------------------------------------------------------------------
    # One full update step
    # ------------------------------------------------------------------

    def update_step(self, omega: Tensor) -> Dict:
        """
        Run one full update: PPO epochs + joint critic update + Lagrangian.

        The joint centralised critic is updated once per PPO epoch to match
        the actor's update frequency.  Previously it was only updated once
        after all PPO epochs, giving the actor n_epochs × N_CELLS × n_batches
        gradient steps while the critic received only n_batches steps — an
        n_epochs × N_CELLS imbalance that caused the value estimates to lag
        behind the rapidly-updated policy, widening the GAE target error.

        Parameters
        ----------
        omega : (N_OBJ,)

        Returns
        -------
        metrics : dict of logged scalars
        """
        # PPO inner loop (actor + cost critics)
        ppo_metrics = self.updater.update(self.buffers, omega)

        # Joint critic update (CTDE) — once per PPO epoch so that the critic
        # sees the same number of passes through the rollout data as the actor.
        jc_loss_sum = 0.0
        for _ in range(self.cfg.n_epochs):
            jc_loss_sum += self._joint_critic_update(omega)
        joint_critic_loss = jc_loss_sum / self.cfg.n_epochs

        # Lagrangian dual update
        mean_costs = {i: self.buffers[i].mean_cost for i in range(N_CELLS)}
        lag_deltas = self.lag_updater.update(mean_costs)
        lambdas    = self.lag_updater.lambdas()

        self.n_updates += 1

        metrics = dict(ppo_metrics)
        metrics["joint_critic_loss"] = joint_critic_loss
        metrics["total_env_steps"]   = self.total_steps
        metrics["n_updates"]         = self.n_updates
        for i in range(N_CELLS):
            metrics[f"lambda_{i}"]          = lambdas[i]
            metrics[f"mean_cost_{i}"]       = mean_costs[i]
            metrics[f"lag_delta_{i}"]       = lag_deltas[i]
            metrics[f"constraint_viol_{i}"] = mean_costs[i] - self.cfg.kappa[i]

        return metrics

    # ------------------------------------------------------------------
    # Evaluation (no gradient)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        omega:     Tensor,
        n_steps:   int   = 500,
    ) -> Dict:
        """
        Run one evaluation episode under a fixed omega (no exploration noise).

        Parameters
        ----------
        omega   : (N_OBJ,)
        n_steps : number of steps to run

        Returns
        -------
        eval_metrics : dict
        """
        omega = omega.to(self.device)
        # Derive a reproducible, omega-specific seed using the same formula
        # as Evaluator._eval_one.  This ensures that evaluating at different
        # omega values produces statistically independent job sequences,
        # making results comparable across preference vectors.  The previous
        # fixed seed (self.cfg.seed + 9999) caused all omega evaluations to
        # share the same job sequence.
        omega_np  = omega.cpu().numpy()
        omega_int = (int(round(omega_np[0] * 1000)) * 1_000_000
                     + int(round(omega_np[1] * 1000)) * 1_000
                     + int(round(omega_np[2] * 1000)))
        # omega_int max ≈ 10^9, safely within numpy.random.default_rng range.
        # No modulo is applied; the previous % 100_000 dropped the o0 component
        # (1_000_000 % 100_000 == 0), causing seed collisions among grid points.
        eval_rng = np.random.default_rng(self.cfg.seed + omega_int)
        eval_env = ManufacturingEnv(
            self.cfg.dataset_path, eval_rng, max_steps=n_steps
        )
        obs_dict = eval_env.reset(omega.cpu().numpy())
        obs_t    = {
            i: torch.as_tensor(obs_dict[i], dtype=torch.float32, device=self.device)
            for i in range(N_CELLS)
        }

        ep_rewards = {i: [] for i in range(N_CELLS)}
        ep_costs   = {i: [] for i in range(N_CELLS)}
        ep_rvecs   = {i: [] for i in range(N_CELLS)}

        thetas, log_stds = self.hnet(omega)

        done = False
        step = 0
        while not done and step < n_steps:
            actions_env = {}
            for i in range(N_CELLS):
                obs_i = obs_t[i].unsqueeze(0)
                mu_i  = PolicyNet.forward(obs_i, thetas[i])
                # Evaluation: use mean action (no noise)
                action_i = torch.clamp(mu_i, -1.0, 1.0)
                actions_env[i] = action_i.squeeze(0).cpu().numpy()

            next_obs, rewards, costs, done, info = eval_env.step(actions_env)

            for i in range(N_CELLS):
                ep_rewards[i].append(rewards[i])
                ep_costs[i].append(float(costs[i]))
                ci = info["cells"][i]
                ep_rvecs[i].append([ci["r1"], ci["r2"], ci["r3"]])
                obs_t[i] = torch.as_tensor(
                    next_obs[i], dtype=torch.float32, device=self.device
                )
            step += 1

        result = {"omega": omega.cpu().tolist(), "n_steps": step}
        for i in range(N_CELLS):
            rv = np.array(ep_rvecs[i])
            result[f"cell{i}_mean_reward"] = float(np.mean(ep_rewards[i]))
            result[f"cell{i}_mean_cost"]   = float(np.mean(ep_costs[i]))
            result[f"cell{i}_mean_r1"]     = float(rv[:, 0].mean())
            result[f"cell{i}_mean_r2"]     = float(rv[:, 1].mean())
            result[f"cell{i}_mean_r3"]     = float(rv[:, 2].mean())
            result[f"cell{i}_feasible"]    = float(
                np.mean(ep_costs[i]) < self.cfg.kappa[i]
            )

        return result

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save(self, tag: str = "latest") -> Path:
        """Save all network weights and Lagrange multipliers."""
        ckpt = {
            "hnet_state":         self.hnet.state_dict(),
            "critic_state":       self.critic.state_dict(),
            "cost_critic_states": [cc.state_dict() for cc in self.cost_critics],
            "lambdas":            self.lag_updater.lambdas(),
            "total_steps":        self.total_steps,
            "n_updates":          self.n_updates,
            "config":             asdict(self.cfg),
        }
        path = self.ckpt_dir / f"ckpt_{tag}.pt"
        torch.save(ckpt, path)
        return path

    def load(self, path: str | Path) -> None:
        """Restore network weights and multipliers from a checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.hnet.load_state_dict(ckpt["hnet_state"])
        self.critic.load_state_dict(ckpt["critic_state"])
        for cc, sd in zip(self.cost_critics, ckpt["cost_critic_states"]):
            cc.load_state_dict(sd)
        for i, lam in ckpt["lambdas"].items():
            self.lag_updater._lambdas[int(i)] = lam
        self.total_steps = ckpt["total_steps"]
        self.n_updates   = ckpt["n_updates"]

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------

    def _open_log(self, fieldnames: List[str]) -> None:
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=fieldnames,
            extrasaction="ignore",
        )
        self._csv_writer.writeheader()

    # Columns kept in the training log (paper-relevant only).
    # Wall-clock time, internal deltas, duplicate cost columns, and the
    # omega dict are excluded — they add no information to the paper.
    _LOG_COLUMNS = [
        "total_env_steps",
        "omega_0", "omega_1", "omega_2",
        "cell0_mean_r1", "cell0_mean_r2", "cell0_mean_r3",
        "cell0_mean_cost", "cell0_constraint_viol",
        "cell1_mean_r1", "cell1_mean_r2", "cell1_mean_r3",
        "cell1_mean_cost", "cell1_constraint_viol",
        "cell2_mean_r1", "cell2_mean_r2", "cell2_mean_r3",
        "cell2_mean_cost", "cell2_constraint_viol",
        "loss_actor", "loss_cost_v", "loss_entropy",
        "loss_div", "clip_frac", "joint_critic_loss",
        "lambda_0", "lambda_1", "lambda_2",
    ]

    def _log_row(self, row: Dict) -> None:
        if self._csv_writer is None:
            self._open_log(self._LOG_COLUMNS)
        filtered = {k: row.get(k, "") for k in self._LOG_COLUMNS}
        self._csv_writer.writerow(filtered)
        self._csv_file.flush()

    def _close_log(self) -> None:
        """Close the CSV log file.  Safe to call more than once."""
        if not self._csv_file.closed:
            self._csv_file.close()

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Run training until total_env_steps is reached.

        Each iteration:
          1. Sample omega.
          2. Collect one rollout of `rollout_steps` steps.
          3. Run PPO + joint critic + Lagrangian update.
          4. Log and checkpoint if scheduled.
        """
        cfg   = self.cfg
        steps_per_rollout = cfg.rollout_steps * N_CELLS   # steps across all cells

        print(f"{'='*60}")
        print(f"  SafePC-MARL Training")
        print(f"{'='*60}")
        print(f"  Device         : {self.device}")
        print(f"  Total steps    : {cfg.total_env_steps:,}")
        print(f"  Rollout steps  : {cfg.rollout_steps} per cell")
        print(f"  PPO epochs     : {cfg.n_epochs}")
        print(f"  Batch size     : {cfg.batch_size}")
        print(f"  Omega mode     : {cfg.omega_mode}")
        print(f"  kappa          : {cfg.kappa}")
        print(f"{'='*60}\n")

        t_start = time.time()

        try:
            while self.total_steps < cfg.total_env_steps:
                # 1. Sample omega for this rollout
                omega = self._sample_omega()

                # 2. Collect rollout
                rollout_info = self.collect_rollout(omega)

                # 3. Update
                update_metrics = self.update_step(omega)

                # 4. Merge metrics for logging
                log_row = {
                    "wall_time":       round(time.time() - t_start, 1),
                    "total_env_steps": self.total_steps,
                    "n_updates":       self.n_updates,
                    **{f"omega_{j}": round(omega[j].item(), 4) for j in range(N_OBJ)},
                    **rollout_info,
                    **{k: round(v, 6) if isinstance(v, float) else v
                       for k, v in update_metrics.items()
                       if k not in ("total_env_steps", "n_updates")},
                }

                if self.n_updates % cfg.log_every == 0:
                    self._log_row(log_row)
                    # Console summary
                    lam_str = "  ".join(
                        f"λ{i}={update_metrics[f'lambda_{i}']:.4f}"
                        for i in range(N_CELLS)
                    )
                    cost_str = "  ".join(
                        f"c{i}={update_metrics[f'mean_cost_{i}']:.4f}"
                        for i in range(N_CELLS)
                    )
                    print(
                        f"[step {self.total_steps:>8,}  upd {self.n_updates:>5}]  "
                        f"actor={update_metrics['loss_actor']:+.4f}  "
                        f"clip={update_metrics['clip_frac']:.3f}  "
                        f"{cost_str}  {lam_str}"
                    )

                # 5. Checkpoint
                if self.n_updates % cfg.save_every == 0:
                    path = self.save(tag=f"step{self.total_steps}")
                    print(f"  Checkpoint saved: {path}")

            # Final checkpoint (only reached if training completed normally)
            final_path = self.save(tag="final")
            elapsed = time.time() - t_start
            print(f"\nTraining complete.  {self.total_steps:,} steps in {elapsed:.1f}s.")
            print(f"Final checkpoint: {final_path}")
            print(f"Log: {self.log_path}")

        finally:
            # Always close the log file, whether training finished normally,
            # was interrupted by an exception, or hit a keyboard interrupt.
            self._close_log()


# ---------------------------------------------------------------------------
# Sanity check  (short smoke-test: 3 updates, not a full training run)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== trainer.py  smoke test (3 updates) ===\n")

    cfg = TrainerConfig(
        dataset_path    = "ai4i2020.xlsx",
        total_env_steps = 500 * N_CELLS * 3,   # exactly 3 rollouts
        rollout_steps   = 500,
        n_epochs        = 2,
        batch_size      = 64,
        omega_mode      = "fixed",
        omega_fixed     = [0.5, 0.3, 0.2],
        log_every       = 1,
        save_every      = 999,
        run_name        = "smoke_test",
        seed            = 0,
    )   # device auto-detected via TrainerConfig default_factory

    trainer = Trainer(cfg)

    # ---- collect_rollout ----
    omega = trainer._sample_omega()
    print(f"omega = {omega.tolist()}")
    t0 = time.time()
    rollout_info = trainer.collect_rollout(omega)
    t1 = time.time()
    print(f"collect_rollout: {t1-t0:.2f}s")
    for k, v in rollout_info.items():
        if k != "omega":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # ---- update_step ----
    t2 = time.time()
    metrics = trainer.update_step(omega)
    t3 = time.time()
    print(f"\nupdate_step: {t3-t2:.2f}s")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    # ---- evaluate ----
    t4 = time.time()
    eval_res = trainer.evaluate(omega, n_steps=200)
    t5 = time.time()
    print(f"\nevaluate (200 steps): {t5-t4:.2f}s")
    for k, v in eval_res.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ---- save / load ----
    path = trainer.save(tag="smoke")
    print(f"\nCheckpoint saved: {path}")
    trainer.load(path)
    print("Checkpoint loaded successfully.")

    # ---- full train (3 updates) ----
    print("\n--- Full train() call (3 updates) ---")
    trainer2 = Trainer(cfg)
    trainer2.train()

    print("\nSmoke test PASSED.")
