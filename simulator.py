"""
simulator.py
============
Three-cell discrete-event manufacturing simulation for SafePC-MARL.

Architecture
------------
ProcessSampler   -- fits and samples from per-variant distributions.
ManufacturingEnv -- Gym-compatible multi-agent environment for 3 cells.

Cell mapping
------------
  Cell 0  <->  Type L  (n=6 000 in dataset, 60 % of arrivals)
  Cell 1  <->  Type M  (n=2 997 in dataset, 30 % of arrivals)
  Cell 2  <->  Type H  (n=1 003 in dataset, 10 % of arrivals)

Observation (per cell)
----------------------
  [T_air_norm, T_proc_norm, omega_norm, tau_norm, wear_norm, omega_adj]
  All features are z-scored using per-variant empirical mean/std.
  omega_adj is the last action taken by this cell (initialised to 0.0).

Action (per cell)
-----------------
  Scalar delta ∈ [-1, 1] representing fractional adjustment to omega_rpm.
  delta is clipped to MAX_DELTA = 0.30 in physical effect.
  Partially-coupled physics (omega full scale, torque half inverse scale):
      omega_eff  = omega_sampled * (1 + delta * MAX_DELTA)
      tau_eff    = tau_sampled   * (1 - delta * MAX_DELTA * 0.5)
      power_eff  = tau_eff * omega_eff * 2π/60   (net increases with delta)
  This creates a genuine r2–r3 trade-off mediated by delta:
      higher delta → higher power  → worse  r2 (energy)
      higher delta → lower  torque → better r3 (ergonomic)
  ω2-weighted agents prefer δ→−1; ω3-weighted agents prefer δ→+1.

Reward vector (shared across cells, then dotted with ω at agent level)
-----------------------------------------------------------------------
  r1  throughput component: (omega_eff - omega_mean[v]) / omega_std[v]
                             (positive when machining speed is above the
                              per-variant mean → higher throughput rate)
  r2  energy component    : -(power_eff - power_mean[v]) / power_std[v]
                             (positive when below average power)
  r3  ergonomic component : -(tau_eff - tau_90[v]) / tau_std[v]
                             (positive when torque is below ergonomic limit)

  Directional tradeoffs created by the action delta:
      higher delta → higher omega_eff → better r1 (faster throughput)
      higher delta → higher power_eff → worse  r2 (worse energy)
      higher delta → lower  tau_eff   → better r3 (better ergonomics)
  ω1- and ω3-weighted agents prefer δ→+1; ω2-weighted agents prefer δ→−1.
  This creates a genuine three-objective structure: the r1-r2 conflict and
  the r2-r3 conflict are both present and mediated by delta.

Safety cost (per cell)
----------------------
  c_i = composite failure proxy {0, 1} computed from effective parameters.
  The episode mean of c_i is compared against kappa_i.

Episode termination
-------------------
  Fixed-horizon: MAX_STEPS steps per episode (default 500).

Seeding
-------
  All randomness flows through a single numpy.random.Generator passed
  at construction time, ensuring reproducibility.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.stats import gaussian_kde
from typing import Dict, List, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safepc_marl.dataset import (
    load_raw, compute_features, per_variant_stats,
    VARIANTS,
    OSF_THRESHOLDS, HDF_DT_LIMIT, HDF_RPM_LIMIT,
    PWF_LOW, PWF_HIGH, TWF_LIMIT,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CELLS:    int   = 3
MAX_DELTA:  float = 0.30          # maximum fractional speed adjustment
MAX_STEPS:  int   = 500           # steps per episode

# Features used in the 4D KDE (tool_wear is sampled independently)
KDE_FEATURES: List[str] = ["T_air", "T_proc", "omega_rpm", "tau"]
OBS_DIM:      int        = 6      # 4 process features (norm) + wear_norm + last_action


# ---------------------------------------------------------------------------
# Job dataclass
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """
    A single manufacturing job drawn from the per-variant distribution.
    All fields are raw (un-normalised) physical values.
    """
    variant:   str
    T_air:     float   # K
    T_proc:    float   # K
    omega_rpm: float   # rpm  (sampled, before agent adjustment)
    tau:       float   # Nm   (sampled, before agent adjustment)
    tool_wear: float   # min


# ---------------------------------------------------------------------------
# ProcessSampler
# ---------------------------------------------------------------------------

class ProcessSampler:
    """
    Fits per-variant distributions from the dataset and generates Jobs.

    Sampling strategy
    -----------------
    (T_air, T_proc, omega_rpm, tau) : multivariate Gaussian KDE (Scott's rule)
        Captures strong empirical correlations:
            T_air ↔ T_proc   (r ≈ +0.88)
            omega_rpm ↔ tau  (r ≈ −0.88)

    tool_wear : bootstrap (empirical resample, independent)
        The feature is nearly Uniform(0, ~200) with a sharp right-truncation
        at the TWF limit and is uncorrelated with all other features (|r| < 0.04).
        KDE overestimates the left-boundary leak (~5 % negative samples);
        bootstrap avoids this entirely.

    Post-sampling corrections
    -------------------------
    - omega_rpm  : rounded to nearest integer, clipped to [1, 3 000]
    - tau        : clipped to [1.0, 100.0] Nm  (physical bounds)
    - tool_wear  : rounded to nearest integer, clipped to [0, 254]
    - T_air      : clipped to [290.0, 310.0] K
    - T_proc     : clipped to [300.0, 320.0] K
    """

    def __init__(self, dataset_path: str, rng: np.random.Generator):
        self._rng = rng

        raw = load_raw(dataset_path)
        df  = compute_features(raw)
        vs  = per_variant_stats(df)

        # Store per-variant KDEs and bootstrap arrays
        self._kdes:      Dict[str, gaussian_kde]   = {}
        self._tw_pool:   Dict[str, np.ndarray]     = {}
        self._feature_stats: Dict[str, Dict]       = {}

        for v in VARIANTS:
            sub = vs[v]["features"]                    # DataFrame, n_v × 5 cols

            # --- 4D KDE ---
            kde_data = sub[KDE_FEATURES].values.T     # shape (4, n_v)
            self._kdes[v] = gaussian_kde(kde_data, bw_method="scott")

            # --- Bootstrap pool for tool_wear ---
            self._tw_pool[v] = sub["tool_wear"].values.astype(np.float64)

            # --- Normalisation statistics (for observations) ---
            full_cols   = KDE_FEATURES + ["tool_wear"]
            emp_mean    = sub[full_cols].mean().to_dict()
            emp_std     = sub[full_cols].std().to_dict()
            self._feature_stats[v] = {
                "mean": emp_mean,
                "std":  emp_std,
                # 90th-percentile torque for ergonomic reward
                "tau_90":   vs[v]["tau_90"],
                "tau_std":  float(sub["tau"].std()),
                # kappa for safety budget
                "kappa":    vs[v]["kappa"],
                # power statistics
                "power_mean": vs[v]["power_mean"],
                "power_std":  vs[v]["power_std"],
            }

    def sample_job(self, variant: str) -> Job:
        """
        Draw one Job from the fitted distributions for the given variant.
        """
        # --- Sample (T_air, T_proc, omega_rpm, tau) from 4D KDE ---
        # gaussian_kde.resample draws columns; we want 1 sample -> shape (4,)
        kde_sample = self._kdes[variant].resample(
            1, seed=int(self._rng.integers(0, 2**31))
        ).flatten()

        T_air, T_proc, omega_rpm, tau = kde_sample

        # --- Sample tool_wear via bootstrap ---
        idx       = self._rng.integers(0, len(self._tw_pool[variant]))
        tool_wear = self._tw_pool[variant][idx]

        # --- Post-sampling physical clipping / rounding ---
        T_air     = float(np.clip(T_air,     290.0,   310.0))
        T_proc    = float(np.clip(T_proc,    300.0,   320.0))
        omega_rpm = float(np.clip(np.round(omega_rpm), 1.0, 3_000.0))
        tau       = float(np.clip(tau,         1.0,   100.0))
        tool_wear = float(np.clip(np.round(tool_wear), 0.0, 254.0))

        return Job(
            variant=variant,
            T_air=T_air,
            T_proc=T_proc,
            omega_rpm=omega_rpm,
            tau=tau,
            tool_wear=tool_wear,
        )

    def normalise(self, variant: str, job: Job) -> np.ndarray:
        """
        Return z-scored observation vector for a job (before appending last_action).
        Shape: (5,)  ->  [T_air_n, T_proc_n, omega_n, tau_n, wear_n]
        """
        stats = self._feature_stats[variant]
        vals  = np.array([
            job.T_air, job.T_proc, job.omega_rpm, job.tau, job.tool_wear
        ])
        keys  = KDE_FEATURES + ["tool_wear"]
        mu    = np.array([stats["mean"][k] for k in keys])
        sigma = np.array([stats["std"][k]  for k in keys])
        return (vals - mu) / (sigma + 1e-8)

    def feature_stats(self, variant: str) -> Dict:
        return self._feature_stats[variant]


# ---------------------------------------------------------------------------
# Failure-proxy helper (operates on effective physical parameters)
# ---------------------------------------------------------------------------

def compute_failure_proxy(
    variant:   str,
    T_air:     float,
    T_proc:    float,
    omega_eff: float,
    tau_eff:   float,
    tool_wear: float,
) -> int:
    """
    Return 1 if any of the four physics-based failure proxies is active,
    else 0.  Inputs are the *effective* (post-agent-adjustment) values.
    """
    power = tau_eff * omega_eff * (2.0 * np.pi / 60.0)

    osf = int(tool_wear * tau_eff > OSF_THRESHOLDS[variant])
    hdf = int((T_proc - T_air) < HDF_DT_LIMIT and omega_eff < HDF_RPM_LIMIT)
    pwf = int(power < PWF_LOW or power > PWF_HIGH)
    twf = int(tool_wear > TWF_LIMIT)

    return int(osf or hdf or pwf or twf)


# ---------------------------------------------------------------------------
# Per-cell step result
# ---------------------------------------------------------------------------

@dataclass
class CellStepResult:
    """All outputs from one agent's action on one job."""
    variant:    str
    job:        Job
    next_job:   Job       # the job sampled for the NEXT step (stored in _current_jobs)
    delta:      float     # applied fractional adjustment ∈ [-MAX_DELTA, MAX_DELTA]
    omega_eff:  float     # effective omega after adjustment
    tau_eff:    float     # effective tau after adjustment
    power_eff:  float     # effective power [W]
    r1:         float     # throughput component (normalised omega_eff)
    r2:         float     # energy component
    r3:         float     # ergonomic component
    cost:       int       # composite failure proxy {0, 1}
    obs_next:   np.ndarray  # next observation vector (OBS_DIM,)


# ---------------------------------------------------------------------------
# ManufacturingEnv
# ---------------------------------------------------------------------------

class ManufacturingEnv:
    """
    Three-cell cell-free manufacturing environment.

    Each cell is an independent Markov decision process sharing the same
    objective weights ω.  Cells act simultaneously (as in CTDE multi-agent RL).

    Public interface
    ----------------
    reset(omega)  -> observations: Dict[int, np.ndarray]
    step(actions) -> (observations, rewards, costs, done, info)

    Parameters (observations, rewards, costs) are dicts keyed by cell index
    {0, 1, 2}.

    Reward convention
    -----------------
    The raw reward vector [r1, r2, r3] is computed per cell step, then
    scalarised by dotting with the operator preference vector ω:
        scalar_reward_i = ω · [r1_i, r2_i, r3_i]
    This is what the agent's policy optimises.

    The unscalarised reward components are returned in info for logging.
    """

    # Cell-to-variant assignment is static (Cell 0→L, Cell 1→M, Cell 2→H).
    # Each cell always processes its own variant; there is no stochastic
    # arrival dispatch across cells.  The dataset fractions (60 / 30 / 10 %)
    # are reflected in the per-variant KDE fits, not in runtime dispatching.

    def __init__(
        self,
        dataset_path: str,
        rng:          np.random.Generator,
        max_steps:    int = MAX_STEPS,
    ):
        self._sampler   = ProcessSampler(dataset_path, rng)
        self._rng       = rng
        self._max_steps = max_steps

        # Cache per-variant stats for reward computation
        self._stats: Dict[str, Dict] = {
            v: self._sampler.feature_stats(v) for v in VARIANTS
        }
        self._kappa: Dict[int, float] = {
            i: self._stats[VARIANTS[i]]["kappa"] for i in range(N_CELLS)
        }

        # State that resets each episode
        self._omega:      np.ndarray             = np.ones(3) / 3.0
        self._step:       int                    = 0
        self._last_action: Dict[int, float]      = {i: 0.0 for i in range(N_CELLS)}
        self._cost_buf:   Dict[int, List[float]] = {i: [] for i in range(N_CELLS)}
        self._current_jobs: Dict[int, Job]       = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def obs_dim(self) -> int:
        return OBS_DIM

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def n_cells(self) -> int:
        return N_CELLS

    @property
    def kappa(self) -> Dict[int, float]:
        """Per-cell safety budget κ_i."""
        return self._kappa

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def reset(
        self, omega: np.ndarray
    ) -> Dict[int, np.ndarray]:
        """
        Reset the environment for a new episode.

        Parameters
        ----------
        omega : array of shape (3,), summing to 1.
            Operator preference vector over [throughput, energy, ergonomics].

        Returns
        -------
        obs : Dict[int, np.ndarray]  — initial observation per cell.
        """
        omega = np.asarray(omega, dtype=np.float64)
        assert omega.shape == (3,),   f"omega must have shape (3,), got {omega.shape}"
        assert abs(omega.sum() - 1.0) < 1e-6, f"omega must sum to 1, got {omega.sum()}"
        assert (omega >= 0).all(),    "omega must be non-negative"

        self._omega       = omega
        self._step        = 0
        self._last_action = {i: 0.0 for i in range(N_CELLS)}
        self._cost_buf    = {i: [] for i in range(N_CELLS)}

        # Sample initial jobs for each cell
        self._current_jobs = {i: self._sampler.sample_job(VARIANTS[i])
                              for i in range(N_CELLS)}

        return {i: self._make_obs(i) for i in range(N_CELLS)}

    def step(
        self, actions: Dict[int, np.ndarray]
    ) -> Tuple[
        Dict[int, np.ndarray],   # next observations
        Dict[int, float],        # scalarised rewards
        Dict[int, int],          # costs (0 or 1)
        bool,                    # done
        Dict,                    # info
    ]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        actions : Dict[int, np.ndarray]
            Continuous action for each cell.  Each value is a 1-D array
            of shape (1,) with values in [-1, 1].

        Returns
        -------
        obs      : next observations
        rewards  : scalarised reward per cell (ω · [r1, r2, r3])
        costs    : composite failure proxy per cell {0, 1}
        done     : True if episode has reached max_steps
        info     : dict with raw reward components, effective params, etc.
        """
        # Guard against calling step() before reset().
        # _current_jobs is populated by reset() and cleared to {} at init,
        # so checking it is both necessary and sufficient.
        assert self._current_jobs, \
            "reset() must be called before the first call to step()."

        obs_next: Dict[int, np.ndarray] = {}
        rewards:  Dict[int, float]      = {}
        costs:    Dict[int, int]        = {}
        info_cells: Dict[int, Dict]     = {}

        for i in range(N_CELLS):
            result = self._process_cell(i, actions[i])
            obs_next[i]   = result.obs_next
            rewards[i]    = float(
                self._omega[0] * result.r1
                + self._omega[1] * result.r2
                + self._omega[2] * result.r3
            )
            costs[i]      = result.cost
            self._cost_buf[i].append(float(result.cost))
            info_cells[i] = {
                "r1": result.r1, "r2": result.r2, "r3": result.r3,
                "cost": result.cost,
                "omega_eff": result.omega_eff,
                "tau_eff":   result.tau_eff,
                "power_eff": result.power_eff,
                "delta":     result.delta,
                "variant":   result.variant,
            }

            # Advance: use the next_job already sampled inside _process_cell
            self._last_action[i]   = float(np.clip(actions[i], -1.0, 1.0).item())
            self._current_jobs[i]  = result.next_job

        self._step += 1
        done = (self._step >= self._max_steps)

        info = {
            "cells":       info_cells,
            "step":        self._step,
            "omega":       self._omega.copy(),
            "cost_means":  {i: np.mean(self._cost_buf[i]) for i in range(N_CELLS)},
        }

        return obs_next, rewards, costs, done, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_cell(self, cell_idx: int, raw_action: np.ndarray) -> CellStepResult:
        """
        Apply one action to the current job of cell_idx and compute all outputs.
        """
        v   = VARIANTS[cell_idx]
        job = self._current_jobs[cell_idx]
        st  = self._stats[v]

        # --- Apply action: partial coupling ---
        # delta controls a speed-torque trade-off with partial coupling:
        #   omega_eff = omega * (1 + delta * MAX_DELTA)         -- speed scales up
        #   tau_eff   = tau   * (1 - delta * MAX_DELTA * 0.5)   -- torque scales down at HALF rate
        #   power_eff = tau_eff * omega_eff * 2π/60             -- net INCREASES with delta
        #                                                           (factor ≈ 1 + 0.5*delta*MAX_DELTA)
        # Three-way trade-off created:
        #   higher delta → higher omega_eff → better r1 (throughput)
        #   higher delta → higher power_eff → worse  r2 (energy)
        #   higher delta → lower  tau_eff   → better r3 (ergonomic)
        # ω1- and ω3-weighted agents prefer δ→+1; ω2-weighted agents prefer δ→−1.
        delta     = float(np.clip(raw_action, -1.0, 1.0).item()) * MAX_DELTA
        omega_eff = float(np.clip(job.omega_rpm * (1.0 + delta),         1.0, 5_000.0))
        tau_eff   = float(np.clip(job.tau       * (1.0 - delta * 0.5),   0.1,   200.0))
        power_eff = tau_eff * omega_eff * (2.0 * np.pi / 60.0)

        # --- Failure cost ---
        cost = compute_failure_proxy(
            v, job.T_air, job.T_proc, omega_eff, tau_eff, job.tool_wear
        )

        # --- Rewards ---
        # r1: throughput — normalised effective machining speed.
        #     Positive when omega_eff is above the per-variant mean, meaning
        #     the cell is processing jobs faster than baseline.
        #     Unlike the previous constant 1.0, this varies with the agent's
        #     action and creates a genuine r1-r2 conflict: higher speed yields
        #     more throughput (r1↑) but consumes more power (r2↓).
        r1 = (omega_eff - st["mean"]["omega_rpm"]) / (st["std"]["omega_rpm"] + 1e-8)

        # r2: energy — negative normalised deviation from variant mean
        #     positive reward when below average power
        r2 = -(power_eff - st["power_mean"]) / (st["power_std"] + 1e-8)

        # r3: ergonomic — negative normalised torque excess above tau_90
        #     positive reward when torque is below the ergonomic limit
        tau_excess = tau_eff - st["tau_90"]
        r3 = -tau_excess / (st["tau_std"] + 1e-8)

        # --- Next job (will become current_jobs[cell_idx] at the next step) ---
        next_job  = self._sampler.sample_job(v)
        obs_next  = self._make_obs_from(cell_idx, next_job,
                                        last_action=float(np.clip(raw_action, -1.0, 1.0).item()))

        return CellStepResult(
            variant=v, job=job, next_job=next_job,
            delta=delta, omega_eff=omega_eff, tau_eff=tau_eff, power_eff=power_eff,
            r1=r1, r2=r2, r3=r3, cost=cost, obs_next=obs_next,
        )

    def _make_obs(self, cell_idx: int) -> np.ndarray:
        """Build observation from current job + last action."""
        return self._make_obs_from(
            cell_idx,
            self._current_jobs[cell_idx],
            self._last_action[cell_idx],
        )

    def _make_obs_from(
        self, cell_idx: int, job: Job, last_action: float
    ) -> np.ndarray:
        """
        Concatenate normalised process features with the last action taken.
        Shape: (OBS_DIM,) = (6,)
        """
        feat    = self._sampler.normalise(VARIANTS[cell_idx], job)  # (5,)
        obs     = np.empty(OBS_DIM, dtype=np.float32)
        obs[:5] = feat.astype(np.float32)
        obs[5]  = np.float32(last_action)
        return obs


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = "ai4i2020.xlsx"
    rng       = np.random.default_rng(0)
    env       = ManufacturingEnv(DATA_PATH, rng, max_steps=500)

    print("=" * 60)
    print("simulator.py  |  sanity check")
    print("=" * 60)
    print(f"obs_dim    = {env.obs_dim}")
    print(f"action_dim = {env.action_dim}")
    print(f"n_cells    = {env.n_cells}")
    print(f"kappa      = {env.kappa}")
    print()

    # --- Run one episode with random actions ---
    omega = np.array([0.5, 0.3, 0.2])
    obs   = env.reset(omega)
    print(f"Initial obs shapes: { {k: v.shape for k, v in obs.items()} }")
    print(f"Sample obs cell 0:  {obs[0]}")
    print()

    ep_rewards  = {i: [] for i in range(N_CELLS)}
    ep_costs    = {i: [] for i in range(N_CELLS)}
    ep_r1, ep_r2, ep_r3 = {i: [] for i in range(N_CELLS)}, \
                           {i: [] for i in range(N_CELLS)}, \
                           {i: [] for i in range(N_CELLS)}

    done = False
    while not done:
        actions = {i: np.array([rng.uniform(-1, 1)]) for i in range(N_CELLS)}
        obs, rewards, costs, done, info = env.step(actions)
        for i in range(N_CELLS):
            ep_rewards[i].append(rewards[i])
            ep_costs[i].append(costs[i])
            ep_r1[i].append(info["cells"][i]["r1"])
            ep_r2[i].append(info["cells"][i]["r2"])
            ep_r3[i].append(info["cells"][i]["r3"])

    print(f"Episode length: {env._step} steps")
    print()
    for i in range(N_CELLS):
        v = VARIANTS[i]
        print(f"Cell {i} (Type {v}):")
        print(f"  mean scalar reward = {np.mean(ep_rewards[i]):.4f}")
        print(f"  mean r1 (throughput) = {np.mean(ep_r1[i]):.4f}  (normalised omega_eff, varies with action)")
        print(f"  mean r2 (energy)   = {np.mean(ep_r2[i]):.4f}")
        print(f"  mean r3 (ergon.)   = {np.mean(ep_r3[i]):.4f}")
        print(f"  mean cost          = {np.mean(ep_costs[i]):.4f}  "
              f"(kappa={env.kappa[i]:.4f})")
        print()

    # --- Verify: cost mean with zero-delta action should ≈ kappa ---
    print("--- Zero-action cost rate (should ≈ empirical kappa) ---")
    obs = env.reset(omega)
    zero_costs = {i: [] for i in range(N_CELLS)}
    done = False
    n_zero = 5_000
    for _ in range(n_zero):
        actions = {i: np.array([0.0]) for i in range(N_CELLS)}
        _, _, costs, done, _ = env.step(actions)
        for i in range(N_CELLS):
            zero_costs[i].append(costs[i])
        if done:
            obs = env.reset(omega)

    for i in range(N_CELLS):
        v    = VARIANTS[i]
        rate = np.mean(zero_costs[i])
        print(f"  Cell {i} (Type {v}): cost_rate={rate:.4f}  kappa={env.kappa[i]:.4f}")
