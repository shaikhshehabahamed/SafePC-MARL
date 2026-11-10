"""
run.py
======
SafePC-MARL  —  single command-line entry point.

Usage
-----
    python run.py                            # full run (4 M steps)
    python run.py --steps 2000000           # explicit step count
    python run.py --smoke                   # 3-rollout smoke test
    python run.py --no-baselines            # skip baselines (Claim 1 skipped)
    python run.py --eval-only path/ckpt.pt  # skip training, load checkpoint

Output layout  runs/<run-name>/
────────────────────────────────────────────────────────────────
  checkpoints/
    ckpt_final.pt              final network weights + lambda values
  results/
    training_log.csv           per-update scalars (loss, cost, lambda, clip)
    pareto_sweep_safepc.csv    45-point simplex sweep (SafePC-MARL)
    pareto_sweep_<bl>.csv      same for each baseline
    bayesian_inference_trials.csv  per-trial per-shift MAE and omega_hat
    claims.json                all numerical claim verdicts
  figures/
    fig1_training_dynamics.{pdf,png}
    fig2_pareto_front.{pdf,png}
    fig3_hypervolume.{pdf,png}
    fig4_simplex_heatmaps.{pdf,png}
    fig5_feasibility.{pdf,png}
    fig6_switching.{pdf,png}
    fig7_bayesian.{pdf,png}
────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import torch

from safepc_marl.trainer            import Trainer, TrainerConfig
from safepc_marl.evaluator          import Evaluator
from safepc_marl.bayesian_inference import BayesianInference
from safepc_marl.baselines          import (
    PPOScalarTrainer, MACPOTrainer, CAPQLTrainer, BaselineConfig,
)
from safepc_marl.networks           import N_CELLS
from safepc_marl.figures            import save_results, plot_all


DATA_PATH = "ai4i2020.xlsx"
KAPPA     = {0: 0.1040, 1: 0.0881, 2: 0.0847}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SafePC-MARL training + evaluation")
    p.add_argument("--data-path",    type=str, default=DATA_PATH,
                   help="Path to ai4i2020.xlsx (default: ai4i2020.xlsx in cwd)")
    p.add_argument("--steps",        type=int, default=4_000_000)
    p.add_argument("--seed",         type=int, default=0)
    p.add_argument("--device",       type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu",
                   help="PyTorch device string (default: cuda if available, else cpu)")
    p.add_argument("--run-name",     type=str, default="safepc_marl")
    p.add_argument("--log-dir",      type=str, default="runs")
    p.add_argument("--eval-steps",   type=int, default=500)
    p.add_argument("--bayes-trials", type=int, default=20)
    p.add_argument("--log-every",    type=int, default=10)
    p.add_argument("--smoke",        action="store_true")
    p.add_argument("--no-baselines", action="store_true")
    p.add_argument("--eval-only",    type=str, default=None)
    return p.parse_args()


def _json_safe(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):  return float(obj)
    if isinstance(obj, np.ndarray):      return obj.tolist()
    if isinstance(obj, dict):            return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):            return [_json_safe(v) for v in obj]
    return obj


def main() -> None:
    args      = _parse_args()
    t_start   = time.time()
    data_path = args.data_path       # user-specified; default = "ai4i2020.xlsx"

    if args.smoke:
        args.steps        = 500 * N_CELLS * 3
        args.eval_steps   = 50
        args.bayes_trials = 3
        args.log_every    = 1

    run_dir = Path(args.log_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # 1. Train SafePC-MARL
    # ----------------------------------------------------------------
    print(f"\n[1/5]  Training SafePC-MARL  ({args.steps:,} steps) ...")

    trainer_cfg = TrainerConfig(
        dataset_path    = data_path,
        total_env_steps = args.steps,
        rollout_steps   = 500,
        n_epochs        = 10,
        batch_size      = 64,
        omega_mode      = "dirichlet",
        kappa           = KAPPA,
        lr_actor        = 2e-4,    # reduced from 3e-4: stability with wider HyperNetwork (256-dim)
        alpha_dual      = 5e-4,    # dual step size: prevents runaway lambda
        alpha_div       = 0.15,    # reduced from 0.20: vectorised 32-sample diversity is inherently stronger
        ent_coef        = 0.02,    # increased from 0.01: more exploration helps simplex coverage
        lambda_init     = 0.02,    # non-zero init: kick-starts constraint enforcement from step 1
        seed            = args.seed,
        device          = args.device,
        log_dir         = args.log_dir,
        run_name        = args.run_name,
        log_every       = args.log_every,
        save_every      = 50,
    )

    trainer = Trainer(trainer_cfg)

    if args.eval_only is not None:
        print(f"       Loading checkpoint: {args.eval_only}")
        trainer.load(args.eval_only)
    else:
        trainer.train()

    hnet    = trainer.hnet
    log_csv = trainer.log_path

    # ----------------------------------------------------------------
    # 2. Train baselines
    # ----------------------------------------------------------------
    baseline_dfs: dict[str, pd.DataFrame] = {}

    if args.no_baselines:
        print("\n[2/5]  Baselines skipped (--no-baselines).")
    else:
        baseline_cfg = BaselineConfig(
            dataset_path    = data_path,
            total_env_steps = args.steps,
            rollout_steps   = 500,
            n_epochs        = 10,
            batch_size      = 64,
            kappa           = KAPPA,
            seed            = args.seed,
            device          = args.device,
            log_every       = args.log_every,
        )
        for key, cls in [("PPO-Scalar", PPOScalarTrainer),
                         ("MACPO",      MACPOTrainer),
                         ("CAPQL",      CAPQLTrainer)]:
            print(f"\n[2/5]  Training baseline: {key} ...")
            bl    = cls(baseline_cfg)
            bl.train(verbose=False)
            baseline_dfs[key] = bl.evaluate(eval_steps=args.eval_steps)

    # ----------------------------------------------------------------
    # 3. Evaluate SafePC-MARL across simplex
    # ----------------------------------------------------------------
    print("\n[3/5]  Evaluating SafePC-MARL across 45-point simplex ...")

    evaluator = Evaluator(
        hnet         = hnet,
        dataset_path = data_path,
        kappa        = KAPPA,
        device       = torch.device(args.device),
        seed         = args.seed,
        eval_steps   = args.eval_steps,
    )
    df_safepc = evaluator.sweep_simplex(verbose=True)

    # Compute SafePC-MARL hypervolume once; reuse the result in claim_metrics.
    hv_result_safepc = evaluator.compute_hypervolume(df_safepc)
    hv_safepc        = hv_result_safepc["hypervolume"]
    hv_baselines: dict[str, float] = {
        name: evaluator.compute_hypervolume(df_bl)["hypervolume"]
        for name, df_bl in baseline_dfs.items()
    }

    # ----------------------------------------------------------------
    # 4. Bayesian inference (Claim 3)
    # ----------------------------------------------------------------
    print(f"\n[4/5]  Bayesian inference ({args.bayes_trials} trials) ...")

    bi = BayesianInference(
        hnet         = hnet,
        dataset_path = data_path,
        device       = torch.device(args.device),
        seed         = args.seed,
    )
    bi.precompute(verbose=False)
    bayes_results = bi.evaluate(n_repeats=args.bayes_trials, verbose=True)

    # ----------------------------------------------------------------
    # 5. Claims, data export, figures
    # ----------------------------------------------------------------
    print("\n[5/5]  Claims, data export, figures ...")

    if baseline_dfs:
        claims = evaluator.claim_metrics(
            df            = df_safepc,
            baseline_hv   = max(hv_baselines.values()),
            bayes_results = bayes_results,
            hv_result     = hv_result_safepc,   # reuse; avoids second HV computation
        )
    else:
        hv_res = hv_result_safepc   # already computed above — no second call needed
        claims = {
            "claim1": {
                "hypervolume":      hv_res["hypervolume"],
                "n_nondominated":   hv_res["n_nondominated"],
                "note":             "Baselines not trained; Claim 1 not evaluated.",
                "claim1_satisfied": None,
            },
            "claim2": evaluator.switching_degradation(df_safepc),
            "claim3": bayes_results,
        }

    df_train = pd.read_csv(log_csv)

    save_results(
        run_dir       = run_dir,
        df_train      = df_train,
        df_safepc     = df_safepc,
        claims        = claims,
        baseline_dfs  = baseline_dfs,
        bayes_results = bayes_results,
        hv_safepc     = hv_safepc,
        hv_baselines  = hv_baselines,
    )

    with open(run_dir / "results" / "claims.json", "w") as f:
        json.dump(_json_safe({
            "run_name":     args.run_name,
            "total_steps":  args.steps,
            "seed":         args.seed,
            "elapsed_s":    round(time.time() - t_start, 1),
            "hv_safepc":    hv_safepc,
            "hv_baselines": hv_baselines,
            "claims":       claims,
        }), f, indent=2)

    plot_all(
        run_dir       = run_dir,
        df_train      = df_train,
        df_safepc     = df_safepc,
        claims        = claims,
        kappa         = KAPPA,
        baseline_dfs  = baseline_dfs,
        bayes_results = bayes_results,
        hv_safepc     = hv_safepc,
        hv_baselines  = hv_baselines,
    )

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    elapsed = time.time() - t_start
    c1 = claims["claim1"]
    c2 = claims["claim2"]
    c3 = claims["claim3"]

    print()
    print("═" * 58)
    print(f"  SafePC-MARL  |  {args.run_name}  |  {elapsed:.0f}s")
    print("═" * 58)
    print(f"  HV (SafePC-MARL)   : {hv_safepc:.6f}")
    if hv_baselines:
        best = max(hv_baselines, key=hv_baselines.get)
        print(f"  HV (best baseline) : {hv_baselines[best]:.6f}  [{best}]")
    if "improvement_pct" in c1:
        print(f"  Claim 1 (HV +15%)  : {c1['improvement_pct']:+.1f}%"
              f"  →  {c1['claim1_satisfied']}")
    else:
        print(f"  Claim 1 (HV +15%)  : {c1.get('note', 'N/A')}")
    print(f"  Claim 2 (switch)   : mean={c2['mean_degradation']*100:.2f}%"
          f"  →  {c2['claim2_satisfied']}")
    print(f"  Claim 3 (Bayes)    : MAE(deploy)={c3['mae_after_shift3']:.4f}"
          f"  MAE(oracle)={c3.get('oracle_mae_after_shift3', float('nan')):.4f}"
          f"  →  {c3['claim3_satisfied']}")
    print("═" * 58)
    print(f"  Outputs  →  {run_dir}/")
    print("═" * 58)


if __name__ == "__main__":
    main()
