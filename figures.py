"""
figures.py
==========
Publication-quality figure generation for the SafePC-MARL paper.

Target journal: Journal of Manufacturing Systems (Elsevier, Q1).
All figures are saved as:
  - PDF  (vector; for LaTeX includegraphics)
  - PNG  (300 DPI; for Word and preview)

Figure dimensions follow Elsevier single/double column guidelines:
  Single column : 88 mm  = 3.46 in
  Double column : 180 mm = 7.09 in

Typeface: Times New Roman (Elsevier standard; available on Windows 11).
Colour palette: Okabe-Ito (2008) — perceptually distinct and safe for
colour-blind readers and greyscale printing.

Figure inventory
----------------
  fig1_training_dynamics .pdf/.png
      Four-panel training dynamics: per-cell safety cost with kappa
      thresholds, Lagrange multipliers, mean reward components, and
      training losses versus environment steps.

  fig2_pareto_front .pdf/.png
      r2 (energy) vs r3 (ergonomic) Pareto front across the 45-point
      preference simplex for SafePC-MARL and all baselines.

  fig3_hypervolume .pdf/.png
      Horizontal bar chart of hypervolume indicators across all methods.

  fig4_simplex_heatmaps .pdf/.png
      Ternary heatmap showing joint constraint feasibility across the
      45-point simplex grid for SafePC-MARL.

  fig5_feasibility .pdf/.png
      Per-cell scatter of rollout safety cost across all 45 omega points,
      sorted by cost magnitude, with kappa thresholds.

  fig6_switching .pdf/.png
      Switching degradation at each test preference vector, with the
      20 % per-point guide; Claim 2 is satisfied if the mean is < 20 %.

  fig7_bayesian .pdf/.png
      MAE vs shift number (mean +/- std across trials) with the 0.12
      Claim 3 threshold line and individual trial trajectories.

Also generates
--------------
  pareto_front_safepc.csv   : Non-dominated objective vectors (SafePC-MARL).
  results_table.csv         : One row per method; all key metrics for LaTeX.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from safepc_marl.evaluator import _nondominated_mask


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

# Elsevier column widths in inches
_COL1 = 3.46
_COL2 = 7.09

# Okabe-Ito palette
_C = {
    "safepc":     "#0072B2",   # blue        — our method
    "ppo_scalar": "#E69F00",   # orange
    "macpo":      "#009E73",   # bluish-green
    "capql":      "#CC79A7",   # reddish-purple
    "cell0":      "#0072B2",   # blue
    "cell1":      "#E69F00",   # orange
    "cell2":      "#009E73",   # bluish-green
    "threshold":  "#000000",   # black
    "neutral":    "#999999",   # grey
}

_MARKERS = {
    "safepc":     "o",
    "ppo_scalar": "s",
    "macpo":      "^",
    "capql":      "D",
}

_METHOD_LABELS = {
    "safepc":     "SafePC-MARL (ours)",
    "ppo_scalar": "PPO-Scalar",
    "macpo":      "MACPO",
    "capql":      "CAPQL",
}

_CELL_LABELS = ["Cell 0 (Type L)", "Cell 1 (Type M)", "Cell 2 (Type H)"]
_KAPPA       = {0: 0.1040, 1: 0.0881, 2: 0.0847}


def _setup_style() -> None:
    """Apply global rcParams for journal-quality figures."""
    plt.rcParams.update({
        # Typeface
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        # Sizes
        "font.size":          9,
        "axes.labelsize":     9,
        "axes.titlesize":     9,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "legend.fontsize":    8,
        "legend.title_fontsize": 8,
        # Layout
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.04,
        # Axes
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "grid.linewidth":     0.4,
        "grid.alpha":         0.35,
        "grid.color":         "#cccccc",
        # Lines
        "lines.linewidth":    1.4,
        "lines.markersize":   5,
        # Legend
        "legend.frameon":     False,
        "legend.borderpad":   0.4,
        # Ticks
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.major.size":   3,
        "ytick.major.size":   3,
    })


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    """Save figure as PDF and PNG."""
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Simplex / ternary helpers
# ---------------------------------------------------------------------------

def _to_cartesian(omega: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert barycentric coordinates (omega1, omega2, omega3) with sum=1
    to 2-D Cartesian coordinates in an equilateral triangle.

    Vertex convention:
      omega1=1 (throughput) -> bottom-left  (0, 0)
      omega2=1 (energy)     -> bottom-right (1, 0)
      omega3=1 (ergonomic)  -> apex         (0.5, sqrt(3)/2)
    """
    omega = np.asarray(omega)
    if omega.ndim == 1:
        omega = omega[np.newaxis, :]
    x = 0.5 * (2.0 * omega[:, 1] + omega[:, 2])
    y = (np.sqrt(3) / 2.0) * omega[:, 2]
    return x, y


def _draw_simplex_frame(ax: plt.Axes) -> None:
    """Draw the equilateral triangle and label the three vertices."""
    vx = [0.0, 1.0, 0.5, 0.0]
    vy = [0.0, 0.0, np.sqrt(3) / 2, 0.0]
    ax.plot(vx, vy, color="black", linewidth=0.8, zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")

    offset = 0.06
    ax.text(-offset, -offset,      r"$\omega_1=1$" + "\n(Throughput)",
            ha="center", va="top",    fontsize=7, style="italic")
    ax.text(1 + offset, -offset,   r"$\omega_2=1$" + "\n(Energy)",
            ha="center", va="top",    fontsize=7, style="italic")
    ax.text(0.5, np.sqrt(3)/2 + offset, r"$\omega_3=1$" + "\n(Ergonomic)",
            ha="center", va="bottom",  fontsize=7, style="italic")


# ---------------------------------------------------------------------------
# Figure 1: Training convergence
# ---------------------------------------------------------------------------

def plot_training(
    df:    pd.DataFrame,
    out_dir: Path,
    kappa: Dict[int, float] = _KAPPA,
) -> None:
    """
    Four-panel training dynamics figure.

    Parameters
    ----------
    df      : training log DataFrame (output of pd.read_csv(training_log.csv))
    out_dir : directory where figures are saved
    kappa   : per-cell safety budgets {0: κ_L, 1: κ_M, 2: κ_H}.
              Defaults to the module-level _KAPPA constant so callers that do
              not pass kappa continue to work, but callers that use non-default
              budgets MUST pass this argument to get correct threshold lines.
    """
    steps = df["total_env_steps"].values / 1e6   # convert to millions

    if len(steps) == 0:
        # No rows logged yet (e.g. log_every > total updates in a smoke test).
        # Write an empty figure rather than crashing on steps[-1].
        fig, axes = plt.subplots(2, 2, figsize=(_COL2, _COL2 * 0.72))
        fig.suptitle("Training dynamics (no data logged)", fontsize=9)
        _save(fig, out_dir, "fig1_training_dynamics")
        return

    fig, axes = plt.subplots(2, 2, figsize=(_COL2, _COL2 * 0.72))
    fig.subplots_adjust(hspace=0.38, wspace=0.32)

    # ---- (a) Per-cell safety cost ----------------------------------------
    ax = axes[0, 0]
    for i in range(3):
        col = f"cell{i}_mean_cost"
        if col in df.columns:
            ax.plot(steps, df[col].values,
                    color=_C[f"cell{i}"], label=_CELL_LABELS[i], zorder=3)
            ax.axhline(kappa[i], color=_C[f"cell{i}"],
                       linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xlabel("Environment steps (×10⁶)")
    ax.set_ylabel("Mean cost $c_i$")
    ax.set_title("(a) Per-cell safety cost")
    ax.legend(loc="upper right", handlelength=1.5)
    # Add kappa annotation for the last threshold line
    ax.text(steps[-1] * 0.98, kappa[0] + 0.002, r"$\kappa_i$",
            fontsize=7, ha="right", color=_C["neutral"], style="italic")

    # ---- (b) Lagrange multipliers ----------------------------------------
    ax = axes[0, 1]
    for i in range(3):
        col = f"lambda_{i}"
        if col in df.columns:
            ax.plot(steps, df[col].values,
                    color=_C[f"cell{i}"], label=_CELL_LABELS[i], zorder=3)
    ax.set_xlabel("Environment steps (×10⁶)")
    ax.set_ylabel(r"Lagrange multiplier $\lambda_i$")
    ax.set_title(r"(b) Lagrange multipliers")
    ax.legend(loc="upper left", handlelength=1.5)

    # ---- (c) Reward components -------------------------------------------
    ax = axes[1, 0]
    # Average r1, r2 and r3 across three cells
    for obj_idx, (label, ls) in enumerate(
            [("r1 (throughput)", ":"), ("r2 (energy)", "-"), ("r3 (ergonomic)", "--")]):
        col_vals = []
        for i in range(3):
            c = f"cell{i}_mean_r{obj_idx + 1}"
            if c in df.columns:
                col_vals.append(df[c].values)
        if col_vals:
            avg = np.mean(col_vals, axis=0)
            color = {0: _C["capql"], 1: _C["safepc"], 2: _C["macpo"]}[obj_idx]
            ax.plot(steps, avg, linestyle=ls, color=color,
                    label=label, zorder=3)
    ax.set_xlabel("Environment steps (×10⁶)")
    ax.set_ylabel("Mean reward component")
    ax.set_title("(c) Reward components (avg. across cells)")
    ax.legend(loc="lower right", handlelength=1.5)

    # ---- (d) Training losses ---------------------------------------------
    ax = axes[1, 1]
    loss_cols = [
        ("loss_actor",  "Actor loss",      _C["safepc"],  "-"),
        ("loss_cost_v", "Cost critic loss", _C["ppo_scalar"], "--"),
        ("loss_div",    "Diversity loss",   _C["macpo"],   ":"),
    ]
    for col, lbl, color, ls in loss_cols:
        if col in df.columns:
            ax.plot(steps, df[col].values,
                    color=color, linestyle=ls, label=lbl, zorder=3)
    ax.set_xlabel("Environment steps (×10⁶)")
    ax.set_ylabel("Loss")
    ax.set_title("(d) Training losses")
    ax.legend(loc="upper right", handlelength=1.5)

    _save(fig, out_dir, "fig1_training_dynamics")


# ---------------------------------------------------------------------------
# Figure 2: Pareto front
# ---------------------------------------------------------------------------

def plot_pareto(
    sweep_dfs: Dict[str, pd.DataFrame],
    out_dir:   Path,
) -> None:
    """
    r2 vs r3 Pareto front for SafePC-MARL and baselines.

    Parameters
    ----------
    sweep_dfs : dict mapping method key (e.g. "safepc", "macpo") to sweep df
    out_dir   : output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(_COL2, _COL2 * 0.42))
    fig.subplots_adjust(wspace=0.30)

    # ---- (a) SafePC-MARL coloured by omega2 (energy preference) ---------
    ax = axes[0]
    if "safepc" in sweep_dfs:
        df     = sweep_dfs["safepc"]
        r2     = df["mean_r2"].values
        r3     = df["mean_r3"].values
        omega2 = df["omega_1"].values          # energy weight
        sc = ax.scatter(r2, r3, c=omega2, cmap="viridis",
                        s=22, zorder=4, edgecolors="none", alpha=0.85)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, shrink=0.85)
        cb.set_label(r"Energy preference $\omega_2$", fontsize=8)
        cb.ax.tick_params(labelsize=7)

        # Highlight non-dominated frontier
        obj   = np.column_stack([r2, r3])
        nd    = _nondominated_mask(obj)
        idx   = np.argsort(r2[nd])
        ax.plot(r2[nd][idx], r3[nd][idx],
                color=_C["safepc"], linewidth=1.2,
                linestyle="-", zorder=5, label="Non-dominated front")
        ax.legend(loc="lower left", handlelength=1.5)

    ax.set_xlabel(r"Mean $r_2$ (energy component)")
    ax.set_ylabel(r"Mean $r_3$ (ergonomic component)")
    ax.set_title("(a) SafePC-MARL — preference colouring")

    # ---- (b) All methods — non-dominated frontiers -----------------------
    ax = axes[1]
    for key, df in sweep_dfs.items():
        r2  = df["mean_r2"].values
        r3  = df["mean_r3"].values
        obj = np.column_stack([r2, r3])
        nd  = _nondominated_mask(obj)
        idx = np.argsort(r2[nd])

        color  = _C.get(key, _C["neutral"])
        marker = _MARKERS.get(key, "o")
        label  = _METHOD_LABELS.get(key, key)

        ax.scatter(r2[nd], r3[nd],
                   color=color, marker=marker,
                   s=20, zorder=4, alpha=0.85)
        ax.plot(r2[nd][idx], r3[nd][idx],
                color=color, linewidth=1.2, label=label, zorder=3)

    ax.set_xlabel(r"Mean $r_2$ (energy component)")
    ax.set_ylabel(r"Mean $r_3$ (ergonomic component)")
    ax.set_title("(b) Non-dominated fronts — all methods")
    ax.legend(loc="lower left", handlelength=1.5, fontsize=7)

    _save(fig, out_dir, "fig2_pareto_front")


# ---------------------------------------------------------------------------
# Figure 3: Hypervolume comparison
# ---------------------------------------------------------------------------

def plot_hypervolume(
    hv_values: Dict[str, float],
    out_dir:   Path,
) -> None:
    """
    Horizontal bar chart of hypervolume per method.

    Parameters
    ----------
    hv_values : dict mapping method key to hypervolume value
    out_dir   : output directory
    """
    # Sort: SafePC-MARL first, then descending HV
    order  = ["safepc"] + sorted(
        [k for k in hv_values if k != "safepc"],
        key=lambda k: hv_values[k], reverse=True
    )
    order  = [k for k in order if k in hv_values]
    labels = [_METHOD_LABELS.get(k, k) for k in order]
    values = [hv_values[k] for k in order]
    colors = [_C.get(k, _C["neutral"]) for k in order]

    if not values:
        # Nothing to plot — hv_values was empty (e.g. hv_safepc is NaN
        # and no baselines were trained).  Write an empty figure so the
        # downstream save_results pipeline does not crash.
        fig, ax = plt.subplots(figsize=(_COL1, _COL1 * 0.65))
        ax.set_title("Hypervolume comparison (no data)")
        _save(fig, out_dir, "fig3_hypervolume")
        return

    fig, ax = plt.subplots(figsize=(_COL1, _COL1 * 0.65))
    bars = ax.barh(range(len(labels)), values, color=colors,
                   edgecolor="white", linewidth=0.5, zorder=3)

    # Value labels
    max_val = max(values)
    for bar, val in zip(bars, values):
        ax.text(val + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="left", fontsize=7)

    # Highlight SafePC-MARL bar
    if "safepc" in order:
        idx = order.index("safepc")
        bars[idx].set_edgecolor(_C["safepc"])
        bars[idx].set_linewidth(1.2)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Hypervolume indicator")
    ax.set_title("Hypervolume comparison")
    ax.set_xlim(0, max_val * 1.18)
    ax.invert_yaxis()

    _save(fig, out_dir, "fig3_hypervolume")


# ---------------------------------------------------------------------------
# Figure 4: Simplex feasibility heatmap
# ---------------------------------------------------------------------------

def plot_simplex_feasibility(
    sweep_df: pd.DataFrame,
    out_dir:  Path,
) -> None:
    """
    Ternary heatmap of joint constraint feasibility across the 45-point grid.

    Parameters
    ----------
    sweep_df : SafePC-MARL sweep DataFrame (output of Evaluator.sweep_simplex)
    out_dir  : output directory
    """
    omega = sweep_df[["omega_0", "omega_1", "omega_2"]].values
    feas  = sweep_df["all_feasible"].values.astype(float)

    x, y = _to_cartesian(omega)

    fig, ax = plt.subplots(figsize=(_COL1, _COL1 * 0.92))
    _draw_simplex_frame(ax)

    sc = ax.tricontourf(x, y, feas, levels=np.linspace(0, 1, 11),
                        cmap="RdYlGn", vmin=0, vmax=1, zorder=2)

    # Overlay scatter points
    ax.scatter(x, y, c=feas, cmap="RdYlGn", vmin=0, vmax=1,
               s=18, edgecolors="white", linewidths=0.3, zorder=3)

    cb = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.04,
                      ticks=[0, 0.25, 0.5, 0.75, 1.0])
    cb.set_label("Joint feasibility rate", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    ax.set_title("Joint constraint feasibility across preference simplex")

    _save(fig, out_dir, "fig4_simplex_heatmaps")


# ---------------------------------------------------------------------------
# Figure 6: Switching degradation
# ---------------------------------------------------------------------------

def plot_switching_degradation(
    sw_results: Dict,
    out_dir:    Path,
) -> None:
    """
    Scatter of runtime-switching degradation per test omega, sorted ascending.

    Parameters
    ----------
    sw_results : output of Evaluator.switching_degradation()
    out_dir    : output directory
    """
    per_point = sw_results["per_point"]
    degs = np.array([p["degradation"] for p in per_point]) * 100   # in %

    sorted_degs = np.sort(degs)
    mean_deg    = float(np.mean(degs))
    n = len(sorted_degs)

    fig, ax = plt.subplots(figsize=(_COL1, _COL1 * 0.72))

    # Colour: green if < 20%, red otherwise
    colors = [_C["macpo"] if d < 20.0 else _C["ppo_scalar"]
              for d in sorted_degs]
    ax.bar(range(n), sorted_degs, color=colors, edgecolor="none",
           width=0.8, zorder=3)

    # 20 % threshold line (per-point visual guide)
    ax.axhline(20.0, color=_C["threshold"], linestyle="--",
               linewidth=1.0, zorder=4)

    # Mean degradation — the actual Claim 2 decision metric.
    # Drawn as a solid line so readers can directly verify the claim.
    mean_color  = _C["safepc"] if mean_deg < 20.0 else _C["ppo_scalar"]
    ax.axhline(mean_deg, color=mean_color, linestyle="-",
               linewidth=1.4, zorder=5)
    ax.text(n * 0.98, mean_deg + 0.5,
            f"mean = {mean_deg:.1f} %",
            ha="right", va="bottom", fontsize=7, color=mean_color)

    # Legend patches
    legend_els = [
        Patch(facecolor=_C["macpo"],      label="Below 20 %"),
        Patch(facecolor=_C["ppo_scalar"], label="At or above 20 %"),
        Line2D([0], [0], color=_C["threshold"], linestyle="--",
               linewidth=1.0, label="20 % per-point guide"),
        Line2D([0], [0], color=mean_color, linestyle="-",
               linewidth=1.4, label="Mean (Claim 2 criterion)"),
    ]
    ax.legend(handles=legend_els, loc="upper left", fontsize=7)

    ax.set_xlabel("Test preference vector index (sorted by degradation)")
    ax.set_ylabel("Switching degradation (%)")
    ax.set_title("Runtime switching degradation (Claim 2: mean < 20 %)")
    ax.set_xlim(-0.5, n - 0.5)

    _save(fig, out_dir, "fig6_switching")


# ---------------------------------------------------------------------------
# Figure 7: Bayesian inference MAE
# ---------------------------------------------------------------------------

def plot_bayesian_inference(
    bayes_results: Dict,
    out_dir:       Path,
) -> None:
    """
    MAE vs shift number with individual trial trajectories and mean ± std.

    Shows both deployment-mode MAE (primary; agent acts under omega_hat) and
    oracle-mode MAE (dashed; agent acts under omega_true) for comparison.

    Parameters
    ----------
    bayes_results : output of BayesianInference.evaluate()
    out_dir       : output directory
    """
    n_shifts     = bayes_results["n_shifts"]
    shifts       = np.arange(1, n_shifts + 1)
    mae_mean     = np.array(bayes_results["mae_per_shift"])
    mae_std      = np.array(bayes_results["std_per_shift"])
    per_trial    = bayes_results["per_trial"]

    # Oracle-mode data is present in results produced by the fixed evaluate().
    # Fall back gracefully if the key is absent (e.g. old checkpoint files).
    oracle_mean = np.array(bayes_results.get("oracle_mae_per_shift", []))
    has_oracle  = oracle_mean.size == n_shifts

    fig, ax = plt.subplots(figsize=(_COL1, _COL1 * 0.75))

    # Individual deployment-mode trials (light)
    for trial in per_trial:
        ax.plot(shifts, trial["mae"], color=_C["safepc"],
                alpha=0.18, linewidth=0.8, zorder=2)

    # Deployment-mode mean ± std (primary metric)
    ax.fill_between(shifts,
                    mae_mean - mae_std,
                    mae_mean + mae_std,
                    color=_C["safepc"], alpha=0.20, zorder=3)
    ax.plot(shifts, mae_mean,
            color=_C["safepc"], linewidth=1.8,
            marker="o", markersize=5, label="Mean MAE (deployment)", zorder=4)

    # Oracle-mode mean (diagnostic upper bound)
    if has_oracle:
        ax.plot(shifts, oracle_mean,
                color=_C["neutral"], linewidth=1.2, linestyle="--",
                marker="s", markersize=4,
                label="Mean MAE (oracle, upper bound)", zorder=3)

    # Claim 3 threshold
    ax.axhline(0.12, color=_C["threshold"], linestyle=":",
               linewidth=1.0, label="Claim 3 threshold (0.12)", zorder=5)

    # Annotation: satisfied / not (based on deployment mode)
    claim_ok = bayes_results["claim3_satisfied"]
    verdict  = "[PASS]" if claim_ok else "[FAIL]"
    color_v  = _C["macpo"] if claim_ok else _C["ppo_scalar"]
    ax.text(n_shifts, mae_mean[-1] + mae_std[-1] * 0.5,
            verdict, ha="right", va="bottom", fontsize=7, color=color_v)

    ax.set_xticks(shifts)
    ax.set_xlabel("Shift number")
    ax.set_ylabel("MAE on preference simplex")
    ax.set_title("Bayesian shift-adaptive inference (Claim 3)")
    ax.legend(loc="upper right", fontsize=7, handlelength=1.5)
    ax.set_xlim(0.7, n_shifts + 0.3)
    ax.set_ylim(bottom=0)

    _save(fig, out_dir, "fig7_bayesian")


# ---------------------------------------------------------------------------
# Figure 5: Per-cell safety cost distribution across the preference simplex
# ---------------------------------------------------------------------------

def plot_constraint_analysis(
    sweep_df: pd.DataFrame,
    out_dir:  Path,
    kappa:    Dict[int, float] = _KAPPA,
) -> None:
    """
    Per-cell safety cost at each of the 45 simplex points, sorted ascending,
    with the empirical kappa threshold drawn as a reference line.

    Purpose: descriptive context showing how cost varies with the preference
    vector at evaluation time. Not a pass/fail verdict.

    Parameters
    ----------
    sweep_df : SafePC-MARL sweep DataFrame
    out_dir  : output directory
    kappa    : per-cell safety budgets {0: κ_L, 1: κ_M, 2: κ_H}.
               Defaults to the module-level _KAPPA constant; callers that
               use non-default budgets MUST pass this argument.
    """
    fig, axes = plt.subplots(1, 3, figsize=(_COL2, _COL2 * 0.38),
                             sharey=False)
    fig.subplots_adjust(wspace=0.30)

    for i, ax in enumerate(axes):
        costs    = np.sort(sweep_df[f"cell{i}_cost"].values)
        n        = len(costs)
        color    = _C[f"cell{i}"]
        kappa_i  = kappa[i]

        # Bar chart of sorted costs — single colour per cell (descriptive, no pass/fail split)
        ax.bar(range(n), costs, color=color, edgecolor="none",
               width=0.9, zorder=3)

        # kappa threshold
        ax.axhline(kappa_i, color=_C["threshold"], linestyle="--",
                   linewidth=1.0, zorder=4)

        # Annotation
        ax.text(n * 0.98, kappa_i * 1.04,
                rf"$\kappa_{i}={kappa_i:.3f}$",
                ha="right", va="bottom", fontsize=7, style="italic")

        ax.set_xlabel(r"$\omega$ index (sorted by cost)")
        ax.set_ylabel(f"Safety cost $c_{i}$")
        ax.set_title(f"({chr(97 + i)}) {_CELL_LABELS[i]}")
        ax.set_xlim(-0.5, n - 0.5)

    _save(fig, out_dir, "fig5_feasibility")


# ---------------------------------------------------------------------------
# Supplementary data files
# ---------------------------------------------------------------------------

def save_pareto_front_csv(
    sweep_df: pd.DataFrame,
    out_dir:  Path,
    filename: str = "pareto_front_safepc.csv",
) -> None:
    """Save non-dominated objective vectors from the SafePC-MARL sweep."""
    obj    = sweep_df[["mean_r1", "mean_r2", "mean_r3"]].values
    nd     = _nondominated_mask(obj)
    nd_df  = sweep_df[nd].reset_index(drop=True)
    nd_df.to_csv(out_dir / filename, index=False)


def save_results_table(
    sweep_dfs:  Dict[str, pd.DataFrame],
    hv_values:  Dict[str, float],
    sw_results: Dict,
    out_dir:    Path,
) -> None:
    """
    Save a summary CSV suitable for copy-paste into a LaTeX table.

    Columns: Method | HV | Cell0_Cost | Cell1_Cost | Cell2_Cost |
             Joint_Feas_Rate | Max_Switch_Deg (%)
    """
    rows = []
    for key, df in sweep_dfs.items():
        row = {
            "Method":           _METHOD_LABELS.get(key, key),
            "Hypervolume":      round(hv_values.get(key, float("nan")), 6),
            "Cell0_Mean_Cost":  round(df["cell0_cost"].mean(),      4),
            "Cell1_Mean_Cost":  round(df["cell1_cost"].mean(),      4),
            "Cell2_Mean_Cost":  round(df["cell2_cost"].mean(),      4),
            "Cell0_Feas_Rate":  round(df["cell0_feasible"].mean(),  4),
            "Cell1_Feas_Rate":  round(df["cell1_feasible"].mean(),  4),
            "Cell2_Feas_Rate":  round(df["cell2_feasible"].mean(),  4),
            "Joint_Feas_Rate":  round(df["all_feasible"].mean(),    4),
        }
        # Switching degradation only computed for SafePC-MARL
        if key == "safepc" and sw_results:
            row["Max_Switch_Deg_pct"]  = round(
                sw_results["max_degradation"]  * 100, 2)
            row["Mean_Switch_Deg_pct"] = round(
                sw_results["mean_degradation"] * 100, 2)
        else:
            row["Max_Switch_Deg_pct"]  = "N/A"
            row["Mean_Switch_Deg_pct"] = "N/A"
        rows.append(row)

    # Put SafePC-MARL first
    key_order = ["safepc", "macpo", "capql", "ppo_scalar"]
    key_list  = [k for k in key_order if k in sweep_dfs]
    key_list += [k for k in sweep_dfs if k not in key_list]

    ordered_rows = {k: None for k in key_list}
    for row in rows:
        label = row["Method"]
        match = [k for k in key_list
                 if _METHOD_LABELS.get(k, k) == label]
        if match:
            ordered_rows[match[0]] = row

    final_rows = [v for v in ordered_rows.values() if v is not None]
    pd.DataFrame(final_rows).to_csv(out_dir / "results_table.csv", index=False)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def save_results(
    run_dir:       str | Path,
    df_train:      pd.DataFrame,
    df_safepc:     pd.DataFrame,
    claims:        Dict,
    baseline_dfs:  Dict[str, pd.DataFrame],
    bayes_results: Dict,
    hv_safepc:     float = float("nan"),
    hv_baselines:  Optional[Dict[str, float]] = None,
) -> None:
    """
    Save all tabular outputs to runs/<run_name>/results/.

    Files written
    -------------
    training_log.csv              — per-update training scalars
    pareto_sweep_safepc.csv       — 45-point simplex sweep (SafePC-MARL)
    pareto_sweep_<baseline>.csv   — same for each baseline
    pareto_front_safepc.csv       — non-dominated objective vectors
    bayesian_inference_trials.csv — per-trial per-shift MAE and omega_hat
    results_table.csv             — summary table for LaTeX

    Parameters
    ----------
    hv_safepc    : SafePC-MARL hypervolume (required for results_table.csv)
    hv_baselines : dict mapping baseline name -> hypervolume, e.g.
                   {"PPO-Scalar": 0.123, "MACPO": 0.098, "CAPQL": 0.105}.
                   Uses the same string keys as baseline_dfs.
    """
    run_dir     = Path(run_dir)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training log
    df_train.to_csv(results_dir / "training_log.csv", index=False)

    # Pareto sweeps
    df_safepc.to_csv(results_dir / "pareto_sweep_safepc.csv", index=False)
    for name, df_bl in baseline_dfs.items():
        safe = name.lower().replace("-", "_").replace(" ", "_")
        df_bl.to_csv(results_dir / f"pareto_sweep_{safe}.csv", index=False)

    # Non-dominated Pareto front
    save_pareto_front_csv(df_safepc, results_dir, "pareto_front_safepc.csv")

    # Bayesian inference trials — one row per (trial, shift)
    per_trial = bayes_results.get("per_trial", [])
    rows = []
    for t, trial in enumerate(per_trial):
        for s in range(len(trial["mae"])):
            rows.append({
                "trial":        t,
                "shift":        s + 1,
                "omega_true_0": trial["omega_true"][0],
                "omega_true_1": trial["omega_true"][1],
                "omega_true_2": trial["omega_true"][2],
                "omega_hat_0":  trial["omega_hat"][s][0],
                "omega_hat_1":  trial["omega_hat"][s][1],
                "omega_hat_2":  trial["omega_hat"][s][2],
                "mae":          trial["mae"][s],
            })
    if rows:
        pd.DataFrame(rows).to_csv(
            results_dir / "bayesian_inference_trials.csv", index=False
        )

    # Results summary table (for LaTeX)
    sweep_dfs_keyed = {"safepc": df_safepc}
    for name, df_bl in baseline_dfs.items():
        sweep_dfs_keyed[name.lower().replace("-", "_").replace(" ", "_")] = df_bl

    # Build hypervolume lookup for every method so results_table.csv is complete.
    # Priority order:
    #   1. hv_safepc argument (most direct)
    #   2. claims["claim1"]["hypervolume"] as a fallback
    #   3. NaN if neither is available
    hv_values: Dict[str, float] = {}
    if not (hv_safepc != hv_safepc):      # True unless hv_safepc is NaN
        hv_values["safepc"] = hv_safepc
    elif "claim1" in claims and "hypervolume" in claims["claim1"]:
        hv_values["safepc"] = claims["claim1"]["hypervolume"]

    if hv_baselines:
        for name, hv in hv_baselines.items():
            key = name.lower().replace("-", "_").replace(" ", "_")
            # Accept both the original name key and the normalised key
            hv_values[key]  = hv
            hv_values[name] = hv

    sw_results = claims.get("claim2", {})
    save_results_table(sweep_dfs_keyed, hv_values, sw_results, results_dir)


def plot_all(
    run_dir:       str | Path,
    df_train:      pd.DataFrame,
    df_safepc:     pd.DataFrame,
    claims:        Dict,
    kappa:         Dict[int, float],
    baseline_dfs:  Dict[str, pd.DataFrame],
    bayes_results: Dict,
    hv_safepc:     float,
    hv_baselines:  Dict[str, float],
) -> None:
    """
    Generate all seven figures and write them to runs/<run_name>/figures/.

    Parameters
    ----------
    run_dir       : experiment root directory
    df_train      : training log DataFrame (from pd.read_csv(training_log.csv))
    df_safepc     : SafePC-MARL 45-point sweep DataFrame
    claims        : output of Evaluator.claim_metrics()
    kappa         : per-cell safety budgets {0: κ_L, 1: κ_M, 2: κ_H}
    baseline_dfs  : dict mapping baseline name → sweep DataFrame
    bayes_results : output of BayesianInference.evaluate()
    hv_safepc     : SafePC-MARL hypervolume
    hv_baselines  : dict mapping baseline name → hypervolume
    """
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    _setup_style()

    # Build unified sweep and HV dicts (safepc first, then baselines)
    sweep_dfs: Dict[str, pd.DataFrame] = {"safepc": df_safepc}
    hv_values: Dict[str, float]        = {"safepc": hv_safepc}
    for name, df_bl in baseline_dfs.items():
        key = name.lower().replace("-", "_").replace(" ", "_")
        sweep_dfs[key] = df_bl
        hv_values[key] = hv_baselines.get(name, hv_baselines.get(key, 0.0))

    # Extract per-claim data.  claims["claim3"] is always the same object
    # as the bayes_results argument (set via claim_metrics()), but prefer
    # the value from claims if present so this function stays self-contained.
    sw_results      = claims.get("claim2", {})
    bayes_results_  = claims.get("claim3", bayes_results)  # does not shadow param

    print("  fig1 — training dynamics")
    plot_training(df_train, fig_dir, kappa=kappa)

    print("  fig2 — Pareto front")
    plot_pareto(sweep_dfs, fig_dir)

    print("  fig3 — hypervolume comparison")
    plot_hypervolume(hv_values, fig_dir)

    print("  fig4 — simplex feasibility heatmap")
    plot_simplex_feasibility(df_safepc, fig_dir)

    print("  fig5 — per-cell safety cost distribution")
    plot_constraint_analysis(df_safepc, fig_dir, kappa=kappa)

    print("  fig6 — switching degradation")
    plot_switching_degradation(sw_results, fig_dir)

    print("  fig7 — Bayesian inference")
    plot_bayesian_inference(bayes_results_, fig_dir)

    print(f"  Figures written to: {fig_dir}/")

