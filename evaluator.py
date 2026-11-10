"""
evaluator.py
============
Evaluation harness for SafePC-MARL.

Implements all metrics needed to assess the three falsifiable claims from
the paper (hypervolume, switching degradation, Bayesian shift inference),
plus general Pareto-front analysis tools.

Public API
----------
Evaluator.sweep_simplex()          -> pd.DataFrame  (45-row results table)
Evaluator.compute_hypervolume()    -> float
Evaluator.constraint_feasibility() -> Dict          (descriptive; used by fig5)
Evaluator.switching_degradation()  -> Dict
Evaluator.claim_metrics()          -> Dict          (Claims 1, 2, 3 in one call)

Simplex grid
------------
45 evenly spaced points with N=8 divisions:
    { (i/8, j/8, k/8) : i+j+k=8, i,j,k >= 0 }
This matches the paper's "9×9 grid projected onto the simplex".

Hypervolume
-----------
The Pareto approximation set P consists of one mean-objective vector per
omega point: each vector is the mean of [r1_cell, r2_cell, r3_cell]
averaged over all three cells and over the evaluation episode.

Since all three objectives are to be maximised, we negate them before
passing to DEAP's minimisation-based hypervolume calculator.

The reference point is set to the nadir (worst) point of P minus a small
margin epsilon = 0.1.  This is computed from the sweep results, not fixed
a priori, so it adapts to the actual scale of the returned objectives.

Switching degradation (Claim 2)
--------------------------------
For each of K test preference vectors omega_k:
  - Compute utility U_hnet(omega_k) = omega_k . [R1, R2, R3] using
    the hypernetwork evaluated at omega_k (zero retraining).
  - Compute utility U_oracle(omega_k) = max_{omega in grid} omega_k . [R1, R2, R3]
    using all OTHER grid points already evaluated.  This oracle represents
    the best any fixed policy in the sweep could achieve for omega_k, which
    upper-bounds a dedicated per-omega trained policy (approximately).
  - Degradation = (U_oracle - U_hnet) / ((|U_oracle| + |U_hnet|) / 2 + epsilon)
The claim requires mean degradation < 20 % across all test points.

Shift-adaptive inference (Claim 3)
------------------------------------
Implemented separately in bayesian_inference.py.
claim_metrics() receives the Bayesian results as a parameter from the caller.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
try:
    # Prefer the compiled C extension (fast WFG algorithm)
    from deap.tools._hypervolume import hv as deap_hv
    _USE_DEAP_C = True
except ImportError:
    # Fall back to moocore (installed as deap ≥1.4 dependency)
    try:
        import moocore as _moocore
        _USE_DEAP_C = False
        _USE_MOOCORE = True
    except ImportError:
        _USE_DEAP_C = False
        _USE_MOOCORE = False

from safepc_marl.networks  import (
    HyperNetwork, PolicyNet, N_CELLS, N_OBJ, OBS_DIM, OMEGA_DIM,
)
from safepc_marl.simulator import ManufacturingEnv, VARIANTS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIMPLEX_DIVISIONS: int   = 8       # N=8 -> 45 grid points
HV_EPSILON:        float = 0.1     # reference-point margin below nadir
EVAL_STEPS:        int   = 500     # steps per omega evaluation episode
SWITCHING_K:       int   = 20      # test omega points for claim 2 (switching degradation)


# ---------------------------------------------------------------------------
# Simplex grid construction
# ---------------------------------------------------------------------------

def build_simplex_grid(n_divisions: int = SIMPLEX_DIVISIONS) -> np.ndarray:
    """
    Return all points (i/N, j/N, k/N) with i+j+k = N, i,j,k >= 0.

    Parameters
    ----------
    n_divisions : int   — N in the formula above

    Returns
    -------
    grid : (n_points, 3) float64 array, all rows sum to 1.
    """
    N   = n_divisions
    pts = []
    for i in range(N + 1):
        for j in range(N + 1 - i):
            k = N - i - j
            pts.append([i / N, j / N, k / N])
    grid = np.array(pts, dtype=np.float64)
    assert grid.shape[0] == (N + 1) * (N + 2) // 2, \
        f"Expected {(N+1)*(N+2)//2} points, got {grid.shape[0]}"
    assert np.allclose(grid.sum(axis=1), 1.0)
    return grid


# ---------------------------------------------------------------------------
# Hypervolume helpers
# ---------------------------------------------------------------------------

def _hypervolume_of_set(
    obj_matrix: np.ndarray,
    reference:  Optional[np.ndarray] = None,
) -> float:
    """
    Compute hypervolume of a set of objective vectors to MAXIMISE.

    Converts to minimisation by negating, then uses DEAP's WFG algorithm.

    Parameters
    ----------
    obj_matrix : (n_points, n_obj)  — each row is one objective vector
    reference  : (n_obj,) reference point in the minimised (negated) space,
                 i.e., nadir of (-obj_matrix) + epsilon.
                 If None, computed automatically from the nadir of obj_matrix.

    Returns
    -------
    hv_value : float
    """
    pts_min = (-obj_matrix).tolist()
    if reference is None:
        # Default: nadir of the negated points (= -ideal of the maximised points)
        # plus a small margin so every point is strictly dominated by the reference.
        nadir_min = np.max(-obj_matrix, axis=0)
        reference = nadir_min + HV_EPSILON
    if _USE_DEAP_C:
        return deap_hv.hypervolume(pts_min, reference.tolist())
    elif _USE_MOOCORE:
        import numpy as _np
        return float(_moocore.hypervolume(
            _np.array(pts_min, dtype=np.float64),
            ref=reference.tolist(),
        ))
    else:
        raise RuntimeError(
            "No hypervolume backend found. Install moocore: pip install moocore"
        )


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Evaluates a trained SafePC-MARL policy across the full preference simplex.

    Parameters
    ----------
    hnet         : HyperNetwork  — trained policy generator
    dataset_path : str
    kappa        : Dict[int, float]  — per-cell safety budgets
    device       : torch.device
    seed         : int
    eval_steps   : int  — episode length for each evaluation point
    """

    def __init__(
        self,
        hnet:         HyperNetwork,
        dataset_path: str,
        kappa:        Dict[int, float],
        device:       torch.device | str | None = None,
        seed:         int  = 0,
        eval_steps:   int  = EVAL_STEPS,
    ):
        self.hnet         = hnet
        self.dataset_path = dataset_path
        self.kappa        = kappa
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device       = torch.device(device)
        self.seed         = seed
        self.eval_steps   = eval_steps
        self._grid        = build_simplex_grid(SIMPLEX_DIVISIONS)

    # ------------------------------------------------------------------
    # Core evaluation: one omega point
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _eval_one(self, omega: np.ndarray) -> Dict:
        """
        Evaluate the policy for one preference vector.

        Returns a dict with mean objectives and costs per cell,
        plus the scalarised utility omega . mean([r1,r2,r3]).
        """
        omega_t = torch.as_tensor(omega, dtype=torch.float32,
                                  device=self.device)

        # Deterministic seed derived from omega components.
        # Using hash() is fragile (PYTHONHASHSEED can randomise string hashes).
        # For floats, hash IS deterministic across CPython processes, but
        # we use an explicit formula to be safe and self-documenting.
        # omega_int encodes all three components: o0 in the millions place,
        # o1 in the thousands place, o2 in the units place.  The maximum
        # value is 1000*1_000_000 = 1_000_000_000, well within the range
        # accepted by numpy.random.default_rng.  The previous code applied
        # % 100_000 which silently discarded the o0 component (because
        # 1_000_000 % 100_000 == 0), causing 15 of the 45 grid points to
        # share seeds — including all three simplex corners sharing seed 0.
        omega_int = int(round(omega[0] * 1000)) * 1_000_000 \
                  + int(round(omega[1] * 1000)) * 1_000 \
                  + int(round(omega[2] * 1000))
        rng = np.random.default_rng(self.seed + omega_int)
        env = ManufacturingEnv(self.dataset_path, rng, max_steps=self.eval_steps)

        obs_dict = env.reset(omega)
        obs_t    = {
            i: torch.as_tensor(obs_dict[i], dtype=torch.float32,
                               device=self.device)
            for i in range(N_CELLS)
        }

        # Get policy parameters (once; omega is fixed for the episode)
        thetas, _ = self.hnet(omega_t)

        ep_rvecs = {i: [] for i in range(N_CELLS)}
        ep_costs = {i: [] for i in range(N_CELLS)}

        done = False
        step = 0
        while not done and step < self.eval_steps:
            actions_env = {}
            for i in range(N_CELLS):
                obs_i    = obs_t[i].unsqueeze(0)
                mu_i     = PolicyNet.forward(obs_i, thetas[i])
                action_i = torch.clamp(mu_i, -1.0, 1.0)
                actions_env[i] = action_i.squeeze(0).cpu().numpy()

            next_obs, _, costs, done, info = env.step(actions_env)

            for i in range(N_CELLS):
                ci = info["cells"][i]
                ep_rvecs[i].append([ci["r1"], ci["r2"], ci["r3"]])
                ep_costs[i].append(float(costs[i]))
                obs_t[i] = torch.as_tensor(
                    next_obs[i], dtype=torch.float32, device=self.device
                )
            step += 1

        # Aggregate per cell
        result: Dict = {
            "omega_0": float(omega[0]),
            "omega_1": float(omega[1]),
            "omega_2": float(omega[2]),
        }

        all_r_means = []
        for i in range(N_CELLS):
            rv = np.array(ep_rvecs[i])     # (T, 3)
            r_mean = rv.mean(axis=0)       # (3,)
            c_mean = np.mean(ep_costs[i])

            result[f"cell{i}_r1"]    = float(r_mean[0])
            result[f"cell{i}_r2"]    = float(r_mean[1])
            result[f"cell{i}_r3"]    = float(r_mean[2])
            result[f"cell{i}_cost"]  = float(c_mean)
            result[f"cell{i}_feasible"] = float(c_mean < self.kappa[i])
            all_r_means.append(r_mean)

        # Mean objective vector across cells (used for Pareto front)
        mean_obj = np.mean(all_r_means, axis=0)   # (3,)
        result["mean_r1"]  = float(mean_obj[0])
        result["mean_r2"]  = float(mean_obj[1])
        result["mean_r3"]  = float(mean_obj[2])

        # Scalarised utility  omega . mean_obj
        result["utility"]  = float(omega @ mean_obj)

        # All cells feasible?
        result["all_feasible"] = float(
            all(np.mean(ep_costs[i]) < self.kappa[i] for i in range(N_CELLS))
        )

        return result

    # ------------------------------------------------------------------
    # Full 45-point sweep
    # ------------------------------------------------------------------

    def sweep_simplex(self, verbose: bool = True) -> pd.DataFrame:
        """
        Evaluate at all 45 simplex grid points.

        Returns
        -------
        df : pd.DataFrame with one row per omega point, 44 columns.
        """
        rows = []
        n    = len(self._grid)

        for idx, omega in enumerate(self._grid):
            if verbose and idx % 10 == 0:
                print(f"  Evaluating point {idx+1}/{n}  omega={omega.round(3)}")
            row = self._eval_one(omega)
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Hypervolume
    # ------------------------------------------------------------------

    def compute_hypervolume(self, df: pd.DataFrame) -> Dict:
        """
        Compute the hypervolume of the Pareto approximation set.

        The set P = { [mean_r1, mean_r2, mean_r3] : row in df }.
        Only non-dominated points in P are used.

        Parameters
        ----------
        df : output of sweep_simplex()

        Returns
        -------
        result : dict with keys:
            "hypervolume"     : float
            "n_points"        : int   — total points in sweep
            "n_nondominated"  : int   — non-dominated subset size
            "reference_point" : list  — reference point used (minimised space)
            "obj_matrix"      : np.ndarray (n_points, 3)
        """
        obj_matrix = df[["mean_r1", "mean_r2", "mean_r3"]].values  # (45, 3)

        # Remove dominated points (maximisation)
        nd_mask = _nondominated_mask(obj_matrix)
        nd_pts  = obj_matrix[nd_mask]

        # Reference point: nadir (negated) + epsilon
        nadir_min = np.max(-nd_pts, axis=0)
        ref_min   = nadir_min + HV_EPSILON

        hv_val = _hypervolume_of_set(nd_pts, reference=ref_min)

        return {
            "hypervolume":     hv_val,
            "n_points":        len(obj_matrix),
            "n_nondominated":  int(nd_mask.sum()),
            "reference_point": ref_min.tolist(),
            "obj_matrix":      obj_matrix,
            "nd_obj_matrix":   nd_pts,
        }

    # ------------------------------------------------------------------
    # Constraint feasibility (descriptive — used by fig5)
    # ------------------------------------------------------------------

    def constraint_feasibility(self, df: pd.DataFrame) -> Dict:
        """
        Compute per-cell and joint constraint feasibility across the sweep.

        Parameters
        ----------
        df : output of sweep_simplex()

        Returns
        -------
        result : dict with:
            "cell{i}_feas_rate"  : fraction of omega points where cell i is feasible
            "joint_feas_rate"    : fraction where ALL cells are feasible
            "cell{i}_mean_cost"  : mean cost across all omega points
            "cell{i}_kappa"      : the budget
            "cell{i}_violation"  : mean_cost - kappa  (negative = satisfied)
        """
        result: Dict = {}
        for i in range(N_CELLS):
            costs  = df[f"cell{i}_cost"].values
            feas   = df[f"cell{i}_feasible"].values
            result[f"cell{i}_feas_rate"]  = float(feas.mean())
            result[f"cell{i}_mean_cost"]  = float(costs.mean())
            result[f"cell{i}_kappa"]      = self.kappa[i]
            result[f"cell{i}_violation"]  = float(costs.mean() - self.kappa[i])

        result["joint_feas_rate"] = float(df["all_feasible"].values.mean())
        return result

    # ------------------------------------------------------------------
    # Switching degradation (Claim 2)
    # ------------------------------------------------------------------

    def switching_degradation(
        self, df: pd.DataFrame, n_test: int = SWITCHING_K
    ) -> Dict:
        """
        Measure runtime-switching degradation.

        For K randomly sampled test preference vectors omega_k:
          U_hnet(omega_k)   = utility achieved by H(omega_k)  [from sweep]
          U_oracle(omega_k) = max_{omega in grid} omega_k · obj(omega)
                              [best utility any of the 45 evaluated policies
                               achieves when applied to omega_k, INCLUDING the
                               policy trained for omega_k itself — this is the
                               tightest feasible upper bound and guarantees
                               degradation ≥ 0 by construction]
          degradation_k = (U_oracle - U_hnet) / ((|U_oracle| + |U_hnet|) / 2 + ε)

        The claim requires mean(degradation_k) < 0.20.

        Parameters
        ----------
        df     : output of sweep_simplex()
        n_test : how many omega points to use as test queries

        Returns
        -------
        result : dict with:
            "max_degradation"     : float  — worst-case switching degradation
            "mean_degradation"    : float  — mean across test points
            "n_test"              : int
            "claim2_satisfied"    : bool   — mean_degradation < 0.20
            "per_point"           : list of dicts (one per test omega)
        """
        obj_matrix = df[["mean_r1", "mean_r2", "mean_r3"]].values   # (45, 3)
        omegas     = df[["omega_0", "omega_1", "omega_2"]].values    # (45, 3)

        # Randomly sample K grid points as test queries
        rng      = np.random.default_rng(self.seed + 1)
        idx_all  = np.arange(len(omegas))
        idx_test = rng.choice(idx_all, size=min(n_test, len(omegas)),
                              replace=False)

        per_point = []
        for idx in idx_test:
            omega_k  = omegas[idx]       # (3,)

            # U_hnet: utility of H(omega_k) evaluated at omega_k
            u_hnet = float(omega_k @ obj_matrix[idx])

            # U_oracle: best utility ANY of the 45 evaluated grid policies
            # achieves when scored against omega_k.  This includes the policy
            # that was evaluated AT omega_k, so u_oracle >= u_hnet always.
            utilities_all = obj_matrix @ omega_k      # (45,)
            u_oracle      = float(utilities_all.max())

            # Degradation: fractional utility gap.
            # Use (|u_oracle| + |u_hnet|) / 2 + eps as the denominator to avoid
            # inflation when utilities are near zero.
            scale = (abs(u_oracle) + abs(u_hnet)) / 2.0 + 1e-8
            deg = (u_oracle - u_hnet) / scale

            per_point.append({
                "omega":       omega_k.tolist(),
                "u_hnet":      u_hnet,
                "u_oracle":    u_oracle,
                "degradation": deg,
            })

        degs = [p["degradation"] for p in per_point]

        return {
            "max_degradation":  float(np.max(degs)),
            "mean_degradation": float(np.mean(degs)),
            "n_test":           len(per_point),
            "claim2_satisfied": bool(np.mean(degs) < 0.20),
            "per_point":        per_point,
        }

    # ------------------------------------------------------------------
    # All claim metrics in one call
    # ------------------------------------------------------------------

    def claim_metrics(
        self,
        df:            pd.DataFrame,
        baseline_hv:   float,
        bayes_results: Dict,
        hv_result:     Optional[Dict] = None,
    ) -> Dict:
        """
        Compute all four falsifiable claim metrics.

        Parameters
        ----------
        df            : output of sweep_simplex()
        baseline_hv   : best baseline hypervolume (required for Claim 1)
        bayes_results : output of BayesianInference.evaluate() (required for Claim 3)
        hv_result     : optional pre-computed output of compute_hypervolume(df).
                        If provided, it is used directly and compute_hypervolume
                        is NOT called again.  Pass this when the caller has already
                        computed the hypervolume (e.g. in run.py) to avoid the
                        redundant computation.

        Returns
        -------
        metrics : nested dict, one key per claim
        """
        metrics: Dict = {}

        # --- Claim 1: Hypervolume >= 15% above best baseline ---
        if hv_result is None:
            hv_result = self.compute_hypervolume(df)
        improvement = (hv_result["hypervolume"] - baseline_hv) / (
            abs(baseline_hv) + 1e-12
        )
        metrics["claim1"] = {
            "hypervolume":         hv_result["hypervolume"],
            "n_nondominated":      hv_result["n_nondominated"],
            "baseline_hv":         baseline_hv,
            "improvement_pct":     float(improvement * 100),
            "claim1_satisfied":    bool(improvement >= 0.15),
        }

        # --- Claim 2: Runtime switching degradation ---
        metrics["claim2"] = self.switching_degradation(df)

        # --- Claim 3: Bayesian shift inference ---
        metrics["claim3"] = bayes_results

        return metrics


# ---------------------------------------------------------------------------
# Non-domination helper
# ---------------------------------------------------------------------------

def _nondominated_mask(obj_matrix: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask of non-dominated points (maximisation).

    Point p dominates q iff p[j] >= q[j] for all j AND p[j] > q[j] for some j.

    Parameters
    ----------
    obj_matrix : (n, n_obj)

    Returns
    -------
    mask : (n,) bool  — True for non-dominated points
    """
    n    = obj_matrix.shape[0]
    mask = np.ones(n, dtype=bool)

    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            # Does j dominate i?
            if (obj_matrix[j] >= obj_matrix[i]).all() and \
               (obj_matrix[j] >  obj_matrix[i]).any():
                mask[i] = False
                break

    return mask


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=== evaluator.py  sanity check ===\n")

    # --- Simplex grid ---
    grid = build_simplex_grid(8)
    print(f"Simplex grid: {len(grid)} points  (expected 45)")
    assert len(grid) == 45

    # --- Hypervolume on known problem ---
    # 3-objective maximisation: points on a sphere approximation
    pts  = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.6, 0.5],   # dominates the axes
    ])
    nadir_min = np.max(-pts, axis=0)
    ref_min   = nadir_min + HV_EPSILON
    hv_val = _hypervolume_of_set(pts, ref_min)
    print(f"Hypervolume (known set): {hv_val:.4f}")
    assert hv_val > 0, "HV must be positive"

    # --- Non-domination ---
    nd = _nondominated_mask(pts)
    print(f"Non-dominated mask: {nd}  (last point should dominate axis points)")
    # [0.6,0.6,0.5] dominates none of the axis points outright in 3D maximisation
    # because axis points have one dimension = 1.0 > 0.6

    # --- Full evaluator smoke test (untrained network, random actions) ---
    from safepc_marl.networks import build_networks

    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nets   = build_networks(device)
    hnet   = nets["hnet"]
    kappa  = {0: 0.104, 1: 0.088, 2: 0.085}

    evaluator = Evaluator(
        hnet         = hnet,
        dataset_path = "ai4i2020.xlsx",
        kappa        = kappa,
        device       = device,
        seed         = 0,
        eval_steps   = 50,       # short for smoke test
    )

    # Evaluate 3 grid points
    print("\nEvaluating 3 simplex points (50 steps each) ...")
    t0 = time.time()
    rows = [evaluator._eval_one(grid[k]) for k in [0, 22, 44]]
    t1  = time.time()
    df_small = pd.DataFrame(rows)
    print(f"  Done in {t1-t0:.2f}s")
    print(f"  Columns: {list(df_small.columns)}")
    print(f"  mean_r1: {df_small['mean_r1'].values}")
    print(f"  utility: {df_small['utility'].values}")

    # HV on small df
    hv_res = evaluator.compute_hypervolume(df_small)
    print(f"\n  Hypervolume (3 pts): {hv_res['hypervolume']:.6f}")
    print(f"  Non-dominated:       {hv_res['n_nondominated']}")

    # Feasibility
    feas = evaluator.constraint_feasibility(df_small)
    print(f"\n  Feasibility:")
    for k, v in feas.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # Switching degradation
    sw = evaluator.switching_degradation(df_small, n_test=3)
    print(f"\n  Switching degradation:")
    print(f"    max:  {sw['max_degradation']:.6f}")
    print(f"    mean: {sw['mean_degradation']:.6f}")
    print(f"    claim2_satisfied: {sw['claim2_satisfied']}")

    # Use the individual methods directly here; claim_metrics needs a baseline_hv
    # which isn't available in this standalone sanity check.
    hv_res = evaluator.compute_hypervolume(df_small)
    feas   = evaluator.constraint_feasibility(df_small)
    sw     = evaluator.switching_degradation(df_small, n_test=3)

    print(f"  claim1 hv:           {hv_res['hypervolume']:.6f}")
    print(f"  feas rate (info):    {feas['joint_feas_rate']:.4f}")
    print(f"  claim2 max_deg:      {sw['max_degradation']:.6f}")
    print(f"  claim3: requires BayesianInference.evaluate() — tested in bayesian_inference.py")

    print("\nSanity check PASSED.")
