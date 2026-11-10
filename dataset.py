"""
dataset.py
==========
Loads the AI4I 2020 Predictive Maintenance Dataset and computes every
empirical constant required by SafePC-MARL.

Public API
----------
load_raw(path)          -> pd.DataFrame   (clean, renamed columns)
compute_features(df)    -> pd.DataFrame   (adds power_W, four proxies,
                                           composite, ergonomic margin)
per_variant_stats(df)   -> dict           (KDE-ready summary per Type)

Nothing in this module trains a model or runs a simulation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Column name map (raw Excel -> internal short names)
# ---------------------------------------------------------------------------
_RENAME = {
    "UDI":                      "UDI",
    "Product ID":               "ProductID",
    "Type":                     "Type",
    "Air temperature [K]":      "T_air",
    "Process temperature [K]":  "T_proc",
    "Rotational speed [rpm]":   "omega_rpm",
    "Torque [Nm]":              "tau",
    "Tool wear [min]":          "tool_wear",
}

# Variant ordering used everywhere (Cell 1 = L, Cell 2 = M, Cell 3 = H)
VARIANTS: list[str] = ["L", "M", "H"]

# ---------------------------------------------------------------------------
# Physics-based failure proxy thresholds (from AI4I methodology / paper)
# ---------------------------------------------------------------------------
# OSF: tool_wear [min] * tau [Nm] exceeds variant-specific limit [Nm·min]
OSF_THRESHOLDS: dict[str, float] = {"L": 11_000.0, "M": 12_000.0, "H": 13_000.0}

# HDF: temperature difference below 8.6 K AND rotational speed below 1 380 rpm
HDF_DT_LIMIT:  float = 8.6     # K
HDF_RPM_LIMIT: float = 1_380.0 # rpm

# PWF: mechanical power outside nominal band [W]
PWF_LOW:  float = 3_500.0  # W
PWF_HIGH: float = 9_000.0  # W

# TWF: cumulative tool wear exceeds limit [min]
TWF_LIMIT: float = 200.0  # min


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Read the Excel file, rename columns to short internal names,
    and assert structural integrity.

    Returns
    -------
    df : pd.DataFrame
        10 000 rows × 8 columns; no nulls; Type ∈ {L, M, H}.
    """
    path = Path(path)
    df = pd.read_excel(path)

    # Validate expected columns
    missing = set(_RENAME.keys()) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")

    df = df.rename(columns=_RENAME)

    # Structural assertions
    assert df.shape == (10_000, 8),         f"Unexpected shape: {df.shape}"
    assert df.isnull().sum().sum() == 0,    "Dataset contains null values"
    assert set(df["Type"].unique()) == {"L", "M", "H"}, \
        f"Unexpected Type values: {df['Type'].unique()}"

    expected_counts = {"L": 6_000, "M": 2_997, "H": 1_003}
    actual_counts   = df["Type"].value_counts().to_dict()
    assert actual_counts == expected_counts, \
        f"Unexpected type counts: {actual_counts}"

    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment the raw dataframe with derived columns:

    power_W   : mechanical power in watts  (tau × omega_rpm × 2π/60)
    OSF       : binary overstrain proxy
    HDF       : binary heat-dissipation proxy
    PWF       : binary power-band proxy
    TWF       : binary tool-wear proxy
    composite : disjunction of all four proxies (overall failure risk)
    tau_excess: signed exceedance of tau above per-variant 90th percentile
                (positive = above threshold → ergonomic risk)

    Parameters
    ----------
    df : output of load_raw()

    Returns
    -------
    df : same dataframe with new columns added (copy).
    """
    df = df.copy()

    # --- Mechanical power ---
    df["power_W"] = df["tau"] * df["omega_rpm"] * (2.0 * np.pi / 60.0)

    # --- OSF proxy ---
    osf_limit = df["Type"].map(OSF_THRESHOLDS)
    df["OSF"] = (df["tool_wear"] * df["tau"] > osf_limit).astype(np.int8)

    # --- HDF proxy ---
    dT = df["T_proc"] - df["T_air"]
    df["HDF"] = ((dT < HDF_DT_LIMIT) & (df["omega_rpm"] < HDF_RPM_LIMIT)).astype(np.int8)

    # --- PWF proxy ---
    df["PWF"] = (
        (df["power_W"] < PWF_LOW) | (df["power_W"] > PWF_HIGH)
    ).astype(np.int8)

    # --- TWF proxy ---
    df["TWF"] = (df["tool_wear"] > TWF_LIMIT).astype(np.int8)

    # --- Composite failure risk ---
    df["composite"] = (
        df["OSF"] | df["HDF"] | df["PWF"] | df["TWF"]
    ).astype(np.int8)

    # --- Ergonomic margin (r3 signal) ---
    # tau_90[v] = 90th-percentile torque for variant v
    tau_90 = df.groupby("Type")["tau"].quantile(0.90)
    df["tau_90"] = df["Type"].map(tau_90)
    # Positive value means torque is ABOVE the ergonomic threshold
    df["tau_excess"] = df["tau"] - df["tau_90"]

    return df


# ---------------------------------------------------------------------------
# Per-variant statistics (used to parameterise the simulation's KDEs)
# ---------------------------------------------------------------------------

def per_variant_stats(df: pd.DataFrame) -> dict:
    """
    Return per-variant descriptive statistics and empirical safety budgets.

    Returns
    -------
    stats : dict keyed by variant ("L", "M", "H"), each value is a dict:
        "n"           : int   — sample size
        "features"    : pd.DataFrame — full feature matrix for KDE fitting
        "tau_90"      : float — 90th-percentile torque [Nm]
        "kappa"       : float — empirical composite proxy rate (κ_i)
        "power_mean"  : float — mean mechanical power [W]
        "power_std"   : float — std  mechanical power [W]
    """
    if "composite" not in df.columns:
        df = compute_features(df)

    feature_cols = ["T_air", "T_proc", "omega_rpm", "tau", "tool_wear"]
    stats: dict = {}

    for v in VARIANTS:
        sub = df[df["Type"] == v].copy()
        stats[v] = {
            "n":           len(sub),
            "features":    sub[feature_cols].reset_index(drop=True),
            "tau_90":      float(sub["tau"].quantile(0.90)),
            "kappa":       float(sub["composite"].mean()),
            "power_mean":  float(sub["power_W"].mean()),
            "power_std":   float(sub["power_W"].std()),
        }

    return stats




# ---------------------------------------------------------------------------
# Quick sanity-check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "ai4i2020.xlsx"
    df   = load_raw(path)
    df   = compute_features(df)
    vs   = per_variant_stats(df)

    print("=" * 60)
    print("SafePC-MARL  |  dataset.py  |  sanity check")
    print("=" * 60)
    print(f"Total rows : {len(df)}")
    print(f"Variants   : {VARIANTS}")
    print()
    for v in VARIANTS:
        s = vs[v]
        print(f"  Cell {VARIANTS.index(v)+1} (Type {v})")
        print(f"    n          = {s['n']}")
        print(f"    tau_90     = {s['tau_90']:.3f} Nm")
        print(f"    kappa (κ)  = {s['kappa']:.4f}")
        print(f"    power mean = {s['power_mean']:.1f} W")
        print(f"    power std  = {s['power_std']:.1f} W")
        print()
    power_mean = float(df["power_W"].mean())
    power_std  = float(df["power_W"].std())
    print(f"Global power mean = {power_mean:.1f} W  (paper: ~6275 W)")
    print(f"Global power std  = {power_std:.1f} W  (paper: ~1060 W)")
    print()
    print("Note: empirical κ values are derived from actual data.")
