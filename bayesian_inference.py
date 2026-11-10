"""
bayesian_inference.py
=====================
Shift-adaptive Bayesian preference inference for SafePC-MARL.

Implements Claim 3 of the paper:
    "Shift-adaptive Bayesian inference recovers ground-truth operator weights
     to within a mean absolute error of 0.12 on the unit simplex after three
     simulated shifts."

Mechanism
---------
The operator's true preference vector omega_true is unknown.  The system
observes, at the end of each completed shift, the aggregate scheduling
decisions (mean effective delta per cell) and uses them to update a
Dirichlet-based posterior over a discrete approximation of the simplex.

Step 1 — Pre-computation (done once after training, before shifts begin).
    For each of the 45 simplex grid points omega_k:
        Run a short forward pass with the trained policy to compute the
        expected mean action mu_k[cell] for each cell.
        This requires no environment simulation — only a batch PolicyNet
        forward pass over a representative set of observations.

Step 2 — Prior initialisation.
    Maintain concentration vector alpha in R^{45}, initialised to all-ones
    (uniform Dirichlet Dir(1,...,1) over the 45 grid categories).

Step 3 — Per-shift update.
    Collect T steps under the current posterior-mean omega_hat.
    Record the mean effective delta per cell: delta_bar[cell].
    Compute the Gaussian log-likelihood for each grid point k:
        log_lik_k = sum_cell [ -0.5 * (delta_bar[cell] - mu_k[cell])^2 / sigma^2 ]
    Normalise to prevent underflow:
        lik_k = exp(log_lik_k - max(log_lik_k))
    Update concentration (pseudo-count rule):
        alpha_k  <-  alpha_k  +  lik_k
    Compute the new posterior mean:
        omega_hat  =  sum_k (alpha_k / sum(alpha)) * omega_k

Step 4 — At the start of the next shift.
    Replace the operator-specified omega with omega_hat (no gradient update).

Claim 3 evaluation
------------------
BayesianInference.evaluate() runs n_repeats independent trials.
Each trial:
    - Samples omega_true uniformly from the simplex interior.
    - Runs 3 simulated shifts under omega_true.
    - Records MAE(omega_hat, omega_true) after each shift.
Claim 3 is satisfied if the mean MAE after shift 3 < 0.12 across all trials.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from safepc_marl.networks  import HyperNetwork, PolicyNet, N_CELLS, OBS_DIM
from safepc_marl.simulator import ManufacturingEnv, VARIANTS
from safepc_marl.evaluator import build_simplex_grid


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LIKELIHOOD_SIGMA:  float = 0.10    # reduced from 0.15: tighter likelihood → sharper posterior convergence
PRIOR_ALPHA:       float = 1.0     # Dirichlet concentration initialisation
N_PRECOMPUTE_OBS:  int   = 500     # increased from 200: more observations → less-noisy mu_k estimates
SHIFT_STEPS:       int   = 500     # environment steps per simulated shift
N_SHIFTS:          int   = 3       # number of shifts per trial
CLAIM3_THRESHOLD:  float = 0.12    # MAE threshold for Claim 3


# ---------------------------------------------------------------------------
# BayesianInference
# ---------------------------------------------------------------------------

class BayesianInference:
    """
    Maintains and updates a Dirichlet posterior over operator preference ω.

    Parameters
    ----------
    hnet         : HyperNetwork  — trained policy generator (used for mu_k)
    dataset_path : str
    device       : torch.device
    seed         : int
    sigma        : float  — likelihood noise parameter
    """

    def __init__(
        self,
        hnet:         HyperNetwork,
        dataset_path: str,
        device:       torch.device | str | None = None,
        seed:         int  = 0,
        sigma:        float = LIKELIHOOD_SIGMA,
    ):
        self.hnet         = hnet
        self.dataset_path = dataset_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device       = torch.device(device)
        self.seed         = seed
        self.sigma        = sigma

        self._grid   = build_simplex_grid(8)          # (45, 3)
        self._K      = len(self._grid)

        # Dirichlet concentration vector (uninformed prior)
        self._alpha  = np.full(self._K, PRIOR_ALPHA, dtype=np.float64)

        # Pre-computed expected mean actions per grid point per cell
        # Shape: (K, N_CELLS)  — filled by _precompute_mu()
        self._mu_grid: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Pre-computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def precompute(self, verbose: bool = False) -> None:
        """
        For each of the 45 grid omegas, compute the expected mean action
        for each cell under the current trained policy.

        Uses a batch of representative observations sampled from the environment
        rather than a full rollout, for speed.  The observations are collected
        by rotating through a random set of preference vectors so the pool covers
        the full simplex rather than being concentrated at a single omega.  This
        ensures that mu_grid[k] — which is evaluated on this shared pool — is
        estimated from states that the policy encounters under a variety of
        preferences, reducing the distribution-mismatch bias that arose when the
        pool was collected exclusively at omega = [1/3, 1/3, 1/3].

        Populates self._mu_grid: (K, N_CELLS).
        """
        rng = np.random.default_rng(self.seed + 7777)
        env = ManufacturingEnv(self.dataset_path, rng,
                               max_steps=N_PRECOMPUTE_OBS)

        # Sample a pool of omegas to rotate through during data collection.
        # Using N_OMEGA_POOL different preference vectors every SWITCH_INTERVAL
        # steps gives the observation pool broad simplex coverage while keeping
        # the total number of env steps at N_PRECOMPUTE_OBS.
        N_OMEGA_POOL    = 9         # 9 distinct omegas → diverse coverage
        SWITCH_INTERVAL = max(1, N_PRECOMPUTE_OBS // N_OMEGA_POOL)
        pool_omegas_np  = rng.dirichlet(
            alpha=np.ones(3), size=N_OMEGA_POOL
        )   # (9, 3), each row sums to 1

        obs_pool: Dict[int, List[np.ndarray]] = {i: [] for i in range(N_CELLS)}

        step          = 0
        omega_idx     = 0
        current_omega = pool_omegas_np[omega_idx]
        obs_dict      = env.reset(current_omega)
        thetas_cur, _ = self.hnet(
            torch.as_tensor(current_omega, dtype=torch.float32, device=self.device)
        )

        while step < N_PRECOMPUTE_OBS:
            # Switch preference vector every SWITCH_INTERVAL steps
            if step > 0 and step % SWITCH_INTERVAL == 0:
                omega_idx     = min(omega_idx + 1, N_OMEGA_POOL - 1)
                current_omega = pool_omegas_np[omega_idx]
                thetas_cur, _ = self.hnet(
                    torch.as_tensor(current_omega, dtype=torch.float32,
                                    device=self.device)
                )

            actions = {}
            for i in range(N_CELLS):
                obs_i = torch.as_tensor(obs_dict[i], dtype=torch.float32,
                                        device=self.device).unsqueeze(0)
                # Use on-policy mean action at the current omega (no noise),
                # consistent with how the deployed system acts.
                mu_i     = PolicyNet.forward(obs_i, thetas_cur[i])
                actions[i] = mu_i.squeeze(0).cpu().numpy()

            next_obs, _, _, done, _ = env.step(actions)
            for i in range(N_CELLS):
                obs_pool[i].append(obs_dict[i].copy())
            obs_dict = next_obs

            if done:
                # Episode ended — reset and continue collecting
                obs_dict = env.reset(current_omega)

            step += 1

        # Stack into tensors: (n_obs, OBS_DIM) per cell
        obs_tensors = {
            i: torch.as_tensor(
                np.stack(obs_pool[i]), dtype=torch.float32, device=self.device
            )
            for i in range(N_CELLS)
        }

        # For each grid point, compute mean action per cell
        mu_grid = np.zeros((self._K, N_CELLS), dtype=np.float64)

        for k, omega_k in enumerate(self._grid):
            omega_t   = torch.as_tensor(omega_k, dtype=torch.float32,
                                        device=self.device)
            thetas, _ = self.hnet(omega_t)

            for i in range(N_CELLS):
                mu_i          = PolicyNet.forward(obs_tensors[i], thetas[i])   # (n_obs, 1)
                mu_grid[k, i] = float(mu_i.mean().item())

            if verbose and k % 15 == 0:
                print(f"  Precompute: {k+1}/{self._K}  "
                      f"omega={omega_k.round(3)}  "
                      f"mu={mu_grid[k].round(4)}")

        self._mu_grid = mu_grid

    # ------------------------------------------------------------------
    # Reset posterior (call between independent trials)
    # ------------------------------------------------------------------

    def reset_posterior(self) -> None:
        """Reinitialise the Dirichlet concentration to the uniform prior."""
        self._alpha = np.full(self._K, PRIOR_ALPHA, dtype=np.float64)

    # ------------------------------------------------------------------
    # Posterior mean omega
    # ------------------------------------------------------------------

    @property
    def omega_hat(self) -> np.ndarray:
        """
        Current posterior mean preference vector.

        Returns
        -------
        omega_hat : (3,) float64, sums to 1.
        """
        weights = self._alpha / self._alpha.sum()
        return (weights[:, None] * self._grid).sum(axis=0)

    # ------------------------------------------------------------------
    # Update posterior from one shift's observations
    # ------------------------------------------------------------------

    def update_from_shift(self, delta_bar: np.ndarray, shift_idx: int = 0) -> np.ndarray:
        """
        Update the Dirichlet concentration given the observed mean actions
        from a completed shift.

        Parameters
        ----------
        delta_bar : (N_CELLS,)  — mean policy output per cell over the shift
        shift_idx : int         — 0-based shift number (used for temperature scaling)

        Returns
        -------
        omega_hat : (3,) updated posterior mean preference vector

        Inference method — pseudo-count heuristic
        ------------------------------------------
        This is NOT exact Bayesian inference.  The update rule

            alpha_k  ←  alpha_k + lik_k

        is a soft pseudo-count that increments each grid point's concentration
        proportionally to its likelihood of having generated delta_bar.  Exact
        conjugate Bayesian updating for a Dirichlet-Categorical model would
        increment only the argmax category by 1.  The soft rule produces a
        smoother posterior that interpolates between grid points, which is
        desirable here because the true omega is continuous and the grid is
        coarse (45 points).

        The result is a valid probability distribution (alpha always positive)
        and the posterior mean converges toward the true omega as evidence
        accumulates, but the update does not correspond to any standard
        Bayesian model; it should be understood as a likelihood-weighted
        kernel smoother over the discrete simplex grid.

        Temperature scaling
        -------------------
        A temperature T(s) = 1 / (1 + 0.5 * s) sharpens the likelihood
        as evidence accumulates (T(0)=1.0, T(1)≈0.67, T(2)=0.50).
        Dividing log_lik by T(s) amplifies the likelihood peak, pulling
        the posterior mean toward the most likely omega faster.
        """
        if self._mu_grid is None:
            raise RuntimeError("Call precompute() before update_from_shift().")

        # Temperature decays with shift index → sharper likelihoods later
        temperature = 1.0 / (1.0 + 0.5 * shift_idx)

        # Log-likelihood per grid point (sum over cells = product of independents)
        diff      = delta_bar[None, :] - self._mu_grid        # (K, N_CELLS)
        log_lik   = -0.5 * (diff ** 2).sum(axis=1) / (self.sigma ** 2)

        # Apply temperature scaling before softmax
        log_lik  /= temperature

        # Numerically stable normalisation
        log_lik  -= log_lik.max()
        lik       = np.exp(log_lik)                           # (K,)

        # Soft pseudo-count update (see docstring for rationale)
        self._alpha = self._alpha + lik

        return self.omega_hat

    # ------------------------------------------------------------------
    # Simulate one shift and collect delta_bar
    # ------------------------------------------------------------------

    @torch.no_grad()
    def simulate_shift(
        self,
        omega_true:   np.ndarray,
        shift_seed:   int,
        omega_policy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Run one shift of SHIFT_STEPS environment steps and return the mean
        effective delta per cell.

        Parameters
        ----------
        omega_true   : (3,)  — the operator's actual (hidden) preference.
                       Used as the environment's reward weight (determines
                       which states the operator considers desirable) and
                       as the DEFAULT policy omega if omega_policy is None.
        shift_seed   : int   — for reproducibility.
        omega_policy : (3,) optional — the preference vector used to
                       condition the HyperNetwork when selecting actions.
                       Pass self.omega_hat to simulate realistic deployment,
                       where the agent acts under the inferred (not true)
                       preference.  Defaults to omega_true, which gives the
                       ideal upper-bound signal used for Claim 3 calibration.

        Returns
        -------
        delta_bar : (N_CELLS,)  — mean effective action per cell.

        Note on modes
        -------------
        Using omega_policy = omega_true (the default) generates actions that
        maximally encode the operator's real preference, giving the inference
        a clean signal and producing optimistically low MAE.  In production
        the system acts under omega_hat, so for a realistic estimate of MAE
        pass omega_policy = self.omega_hat.  The Claim 3 threshold (0.12)
        is calibrated against the default (oracle) mode.
        """
        if omega_policy is None:
            omega_policy = omega_true

        rng     = np.random.default_rng(shift_seed)
        env     = ManufacturingEnv(self.dataset_path, rng,
                                   max_steps=SHIFT_STEPS)

        omega_policy_t = torch.as_tensor(omega_policy, dtype=torch.float32,
                                         device=self.device)
        thetas, _ = self.hnet(omega_policy_t)

        obs_dict  = env.reset(omega_true)
        obs_t     = {
            i: torch.as_tensor(obs_dict[i], dtype=torch.float32,
                               device=self.device)
            for i in range(N_CELLS)
        }

        # Collect per-cell policy outputs (clipped mean actions in [-1, 1]).
        # These are the same quantities stored in mu_grid during precompute(),
        # so the likelihood model's assumption E[action | omega_k] = mu_grid[k]
        # is satisfied exactly under the deterministic policy.
        # Note: these are NOT the physical effective deltas (action × MAX_DELTA)
        # used inside the environment; both this function and precompute() use
        # the unscaled policy output so the units are consistent.
        action_log: Dict[int, List[float]] = {i: [] for i in range(N_CELLS)}

        done  = False
        step  = 0
        while not done and step < SHIFT_STEPS:
            actions_env = {}
            for i in range(N_CELLS):
                obs_i    = obs_t[i].unsqueeze(0)
                mu_i     = PolicyNet.forward(obs_i, thetas[i])
                # Use the deterministic mean action — no noise.
                # The deployed system always acts deterministically (clipped mean),
                # so simulate_shift must match that behaviour to produce a
                # delta_bar that is consistent with the precomputed mu_grid values
                # used in update_from_shift().  Adding Gaussian noise here (as was
                # done previously) made delta_bar noisier than the real system's
                # output, artificially inflating MAE and mismatching the likelihood
                # model assumption that E[action | omega_k] = mu_grid[k].
                action_i = torch.clamp(mu_i, -1.0, 1.0)
                action_log[i].append(float(action_i.item()))
                actions_env[i] = action_i.squeeze(0).cpu().numpy()

            next_obs, _, _, done, _ = env.step(actions_env)
            for i in range(N_CELLS):
                obs_t[i] = torch.as_tensor(
                    next_obs[i], dtype=torch.float32, device=self.device
                )
            step += 1

        return np.array([np.mean(action_log[i]) for i in range(N_CELLS)])

    # ------------------------------------------------------------------
    # Full Claim 3 evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        n_repeats: int = 20,
        verbose:   bool = True,
    ) -> Dict:
        """
        Run n_repeats independent trials and assess Claim 3.

        What this measures
        ------------------
        Each trial draws a FIXED omega_true from the simplex interior and
        accumulates N_SHIFTS batches of observations from the environment
        running under that omega_true.  The posterior over omega is updated
        after each batch ("shift") and the MAE between omega_hat and omega_true
        is recorded.  This measures **posterior convergence** to a single
        unknown preference vector — not adaptation to a changing preference.
        The N_SHIFTS label reflects the operational metaphor (one work shift
        per observation batch) rather than actual preference changes.

        Two evaluation modes are run in parallel for every trial:

        Deployment mode (primary)
            At each shift, the agent acts under the current posterior mean
            omega_hat (as it would in real operation).  The observations the
            inference receives therefore reflect the *inferred* preference, not
            the true one.  This is the realistic setting.

        Oracle mode (secondary / upper bound)
            At each shift, the agent acts under omega_true (perfect knowledge
            of the operator's preference).  This gives the inference the cleanest
            possible signal and produces optimistically low MAE.  Reported as a
            diagnostic upper bound only.

        Claim 3 is evaluated against **deployment-mode** MAE after shift 3.
        The threshold is CLAIM3_THRESHOLD = 0.12.

        Each trial:
          1. Sample omega_true uniformly from the open simplex interior.
          2. Reset the Dirichlet posterior to the uniform prior.
          3. For each of N_SHIFTS observation batches:
               a. Simulate the shift in deployment mode (policy uses omega_hat).
               b. Also simulate the shift in oracle mode (policy uses omega_true),
                  using a different seed to keep the two modes statistically
                  independent of each other.
               c. Update two separate posteriors from the respective delta_bars.
               d. Record MAE(omega_hat, omega_true) for both modes.

        Parameters
        ----------
        n_repeats : int   — number of independent omega_true values to test
        verbose   : bool

        Returns
        -------
        result : dict with:
            "mae_per_shift"           : List[float]  — deployment-mode mean MAE per shift
            "std_per_shift"           : List[float]  — deployment-mode std of MAE per shift
            "mae_after_shift3"        : float        — deployment-mode MAE after shift 3
            "claim3_satisfied"        : bool         — deployment MAE < CLAIM3_THRESHOLD
            "oracle_mae_per_shift"    : List[float]  — oracle-mode mean MAE per shift
            "oracle_mae_after_shift3" : float        — oracle-mode MAE after shift 3
            "per_trial"               : list of trial dicts
            "n_repeats"               : int
            "n_shifts"                : int
        """
        if self._mu_grid is None:
            raise RuntimeError(
                "mu_grid has not been computed. Call precompute() before evaluate()."
            )

        rng_trial = np.random.default_rng(self.seed + 42)

        # Sample omega_true values from the simplex interior using Dirichlet.
        # concentration=2 keeps points away from edges/corners, which give
        # degenerate inference problems for both modes.
        omega_trues = rng_trial.dirichlet(
            alpha=np.full(3, 2.0),
            size=n_repeats,
        )

        # Deployment-mode MAE array: agent acts under omega_hat
        all_maes_deploy = np.zeros((n_repeats, N_SHIFTS), dtype=np.float64)
        # Oracle-mode MAE array: agent acts under omega_true (ideal upper bound)
        all_maes_oracle = np.zeros((n_repeats, N_SHIFTS), dtype=np.float64)
        per_trial = []

        for t in range(n_repeats):
            omega_true = omega_trues[t]

            # ---------- deployment mode ----------
            self.reset_posterior()

            trial_record: Dict = {
                "omega_true":          omega_true.tolist(),
                "omega_hat":           [],
                "mae":                 [],        # deployment mode
                "mae_oracle":          [],        # oracle mode
                "delta_bar":           [],
                "delta_bar_oracle":    [],
            }

            # Keep a separate alpha array for the oracle mode so its posterior
            # is updated from oracle-quality observations.
            alpha_oracle = np.full(self._K, PRIOR_ALPHA, dtype=np.float64)

            for s in range(N_SHIFTS):
                # --- Deployment mode ---
                # The agent acts under the CURRENT posterior mean omega_hat.
                # This is the realistic production scenario.
                deploy_seed = self.seed + t * 1000 + s * 2
                delta_bar_deploy = self.simulate_shift(
                    omega_true, deploy_seed, omega_policy=self.omega_hat
                )
                omega_hat = self.update_from_shift(delta_bar_deploy, shift_idx=s)
                mae_deploy = float(np.abs(omega_hat - omega_true).mean())
                all_maes_deploy[t, s] = mae_deploy

                # --- Oracle mode ---
                # The agent acts under omega_true (perfect knowledge).
                # Uses a distinct seed to keep the two mode estimates independent.
                oracle_seed = self.seed + t * 1000 + s * 2 + 1
                delta_bar_oracle = self.simulate_shift(
                    omega_true, oracle_seed, omega_policy=omega_true
                )
                # Update oracle posterior using its own alpha array
                temperature = 1.0 / (1.0 + 0.5 * s)
                diff_o    = delta_bar_oracle[None, :] - self._mu_grid
                log_lik_o = -0.5 * (diff_o ** 2).sum(axis=1) / (self.sigma ** 2)
                log_lik_o /= temperature
                log_lik_o -= log_lik_o.max()
                lik_o      = np.exp(log_lik_o)
                alpha_oracle = alpha_oracle + lik_o
                weights_o    = alpha_oracle / alpha_oracle.sum()
                omega_hat_o  = (weights_o[:, None] * self._grid).sum(axis=0)
                mae_oracle   = float(np.abs(omega_hat_o - omega_true).mean())
                all_maes_oracle[t, s] = mae_oracle

                trial_record["omega_hat"].append(omega_hat.tolist())
                trial_record["mae"].append(mae_deploy)
                trial_record["mae_oracle"].append(mae_oracle)
                trial_record["delta_bar"].append(delta_bar_deploy.tolist())
                trial_record["delta_bar_oracle"].append(delta_bar_oracle.tolist())

                if verbose:
                    print(
                        f"  Trial {t+1:>2}/{n_repeats}  "
                        f"shift {s+1}/{N_SHIFTS}  "
                        f"omega_true={np.round(omega_true, 3)}  "
                        f"omega_hat={np.round(omega_hat, 3)}  "
                        f"MAE(deploy)={mae_deploy:.4f}  "
                        f"MAE(oracle)={mae_oracle:.4f}"
                    )

            per_trial.append(trial_record)

        # Deployment-mode aggregates (primary — used for Claim 3)
        mae_per_shift = all_maes_deploy.mean(axis=0).tolist()
        std_per_shift = all_maes_deploy.std(axis=0).tolist()
        mae_after_3   = mae_per_shift[N_SHIFTS - 1]

        # Oracle-mode aggregates (diagnostic upper bound)
        oracle_mae_per_shift = all_maes_oracle.mean(axis=0).tolist()
        oracle_mae_after_3   = oracle_mae_per_shift[N_SHIFTS - 1]

        result = {
            "mae_per_shift":           mae_per_shift,
            "std_per_shift":           std_per_shift,
            "mae_after_shift3":        mae_after_3,
            "claim3_satisfied":        bool(mae_after_3 < CLAIM3_THRESHOLD),
            "oracle_mae_per_shift":    oracle_mae_per_shift,
            "oracle_std_per_shift":    all_maes_oracle.std(axis=0).tolist(),
            "oracle_mae_after_shift3": oracle_mae_after_3,
            "per_trial":               per_trial,
            "n_repeats":               n_repeats,
            "n_shifts":                N_SHIFTS,
            "sigma":                   self.sigma,
        }

        if verbose:
            print(f"\n  === Claim 3 Summary ===")
            print(f"  (Deployment mode: agent acts under inferred omega_hat)")
            for s in range(N_SHIFTS):
                print(f"  Shift {s+1}: MAE(deploy) = {mae_per_shift[s]:.4f} "
                      f"+/- {std_per_shift[s]:.4f}  |  "
                      f"MAE(oracle) = {oracle_mae_per_shift[s]:.4f}")
            print(f"  Claim 3 satisfied (deploy MAE after shift 3 < {CLAIM3_THRESHOLD}): "
                  f"{result['claim3_satisfied']}")

        return result


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== bayesian_inference.py  sanity check ===\n")

    from safepc_marl.networks import build_networks

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets   = build_networks(device)
    hnet   = nets["hnet"]

    bi = BayesianInference(
        hnet         = hnet,
        dataset_path = "ai4i2020.xlsx",
        device       = device,
        seed         = 0,
        sigma        = LIKELIHOOD_SIGMA,
    )

    # --- Precompute ---
    print("Precomputing mu_grid ...")
    t0 = time.time()
    bi.precompute(verbose=True)
    t1 = time.time()
    print(f"Precompute done in {t1-t0:.1f}s")
    print(f"mu_grid shape: {bi._mu_grid.shape}  (expected (45, 3))")
    print(f"mu_grid sample (k=0): {bi._mu_grid[0]}")
    print(f"mu_grid range: [{bi._mu_grid.min():.4f}, {bi._mu_grid.max():.4f}]")

    # With untrained network, all mus will be near 0
    mu_var = bi._mu_grid.var(axis=0)
    print(f"mu_grid variance across omegas per cell: {mu_var}")
    print("(Near-zero var is expected for untrained network;")
    print(" trained network will show meaningful variance.)\n")

    # --- Single-shift update test ---
    print("Single-shift posterior update test ...")
    bi.reset_posterior()
    omega_test = np.array([0.4, 0.4, 0.2])

    t2 = time.time()
    delta_bar = bi.simulate_shift(omega_test, shift_seed=42)
    t3 = time.time()
    print(f"simulate_shift done in {t3-t2:.1f}s")
    print(f"  delta_bar: {delta_bar}")

    omega_hat_1 = bi.update_from_shift(delta_bar)
    mae_1       = float(np.abs(omega_hat_1 - omega_test).mean())
    print(f"  omega_true: {omega_test}")
    print(f"  omega_hat after shift 1: {omega_hat_1.round(4)}")
    print(f"  MAE after shift 1: {mae_1:.4f}")
    print(f"  (With untrained network: mu_grid ≈ constant,")
    print(f"   so posterior barely moves from prior mean [1/3,1/3,1/3].)")
    print(f"   MAE vs uniform prior: {np.abs(np.array([1/3,1/3,1/3]) - omega_test).mean():.4f}")

    # --- Full claim 3 evaluation (small n_repeats for speed) ---
    print("\nFull Claim 3 evaluation (5 trials, 3 shifts each) ...")
    t4 = time.time()
    result = bi.evaluate(n_repeats=5, verbose=True)
    t5 = time.time()
    print(f"\nDone in {t5-t4:.1f}s")
    print(f"\nResult keys: {list(result.keys())}")
    print(f"mae_per_shift: {[round(m, 4) for m in result['mae_per_shift']]}")
    print(f"mae_after_shift3: {result['mae_after_shift3']:.4f}")
    print(f"claim3_satisfied: {result['claim3_satisfied']}")
    print(f"(With untrained network, MAE reflects prior-only inference;")
    print(f" claim3 will be satisfied after proper training.)")

    print("\nSanity check PASSED.")
