#!/usr/bin/env python
"""
Diagnose alternative uncorrected/corrected error definitions for beta calibration.

Usage:
  python scripts/diagnose_beta_errors.py --csv results/beta_intermediate/beta_validity_n15000_T50000.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=str,
        default="results/beta_intermediate/beta_validity_n15000_T50000.csv",
    )
    args = ap.parse_args()

    path = Path(args.csv)
    df = pd.read_csv(path)

    p = df["p"].to_numpy(float)
    lam = df["lam"].to_numpy(float)
    C = df["C"].to_numpy(float)
    X0 = df["X0"].to_numpy(float)
    eta_hat = df["eta_hat"].to_numpy(float)
    beta = df["beta_implied"].to_numpy(float)

    A = p * (1.0 - p) * C ** 2

    sigma_hat = X0 ** 2 / eta_hat
    sigma_unc_with_lam = lam + A
    sigma_unc_no_lam = A

    eta_unc_with_lam = X0 ** 2 / sigma_unc_with_lam
    eta_unc_no_lam = X0 ** 2 / sigma_unc_no_lam

    def mae(name, arr):
        print(f"{name:45s}: {float(np.mean(arr)):.4f}%")

    print("=" * 72)
    print(f"File: {path}")
    print(f"Rows: {len(df)}")
    print("=" * 72)

    print("\nETA error, denominator eta_hat")
    mae(
        "uncorrected with lambda",
        100.0 * np.abs(eta_unc_with_lam - eta_hat) / eta_hat,
    )
    mae(
        "uncorrected no lambda",
        100.0 * np.abs(eta_unc_no_lam - eta_hat) / eta_hat,
    )

    print("\nETA error, denominator eta_unc")
    mae(
        "uncorrected with lambda",
        100.0 * np.abs(eta_unc_with_lam - eta_hat) / eta_unc_with_lam,
    )
    mae(
        "uncorrected no lambda",
        100.0 * np.abs(eta_unc_no_lam - eta_hat) / eta_unc_no_lam,
    )

    print("\nSIGMA2 error")
    mae(
        "sigma unc with lambda / sigma_hat denom",
        100.0 * np.abs(sigma_unc_with_lam - sigma_hat) / sigma_hat,
    )
    mae(
        "sigma unc with lambda / sigma_unc denom",
        100.0 * np.abs(sigma_unc_with_lam - sigma_hat) / sigma_unc_with_lam,
    )
    mae(
        "sigma unc no lambda / sigma_hat denom",
        100.0 * np.abs(sigma_unc_no_lam - sigma_hat) / sigma_hat,
    )
    mae(
        "sigma unc no lambda / sigma_unc denom",
        100.0 * np.abs(sigma_unc_no_lam - sigma_hat) / sigma_unc_no_lam,
    )

    print("\nBETA error relative to beta=1")
    mae(
        "abs(beta - 1) / beta",
        100.0 * np.abs(beta - 1.0) / beta,
    )
    mae(
        "abs(beta - 1) / 1",
        100.0 * np.abs(beta - 1.0),
    )

    print("\nSubgroups")
    for label, mask in [
        ("p <= 0.10", p <= 0.10),
        ("0.10 < p <= 0.20", (p > 0.10) & (p <= 0.20)),
        ("0.20 < p <= 0.35", (p > 0.20) & (p <= 0.35)),
        ("delta <= 3", df["delta"].to_numpy(float) <= 3.0),
        ("3 < delta <= 7", (df["delta"].to_numpy(float) > 3.0) & (df["delta"].to_numpy(float) <= 7.0)),
        ("7 < delta <= 10", (df["delta"].to_numpy(float) > 7.0) & (df["delta"].to_numpy(float) <= 10.0)),
        ("10 < delta <= 15", (df["delta"].to_numpy(float) > 10.0) & (df["delta"].to_numpy(float) <= 15.0)),
    ]:
        if mask.sum() == 0:
            continue
        err = 100.0 * np.abs(eta_unc_with_lam[mask] - eta_hat[mask]) / eta_hat[mask]
        print(f"{label:20s}: n={int(mask.sum()):2d}, MAE_unc={float(err.mean()):.4f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
