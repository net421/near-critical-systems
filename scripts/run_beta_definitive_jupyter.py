#!/usr/bin/env python
"""Standalone, resumable beta calibration for a Jupyter web notebook.

Jupyter usage:
    %pip install numpy scipy pandas numba
    !python -u beta_calibration_jupyter.py --threads 16

The defaults reproduce the definitive manuscript protocol and print every
result directly in the notebook. Add --save only if checkpoint/output files
are desired. The script only
simulates the 42 canonical validity-window points; the original pipeline also
simulated 44 duplicate/out-of-window points that never entered the fit.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from numba import get_num_threads, njit, prange, set_num_threads
from scipy.optimize import curve_fit


# Edit these values when pasting the script directly into a Jupyter cell.
JUPYTER_THREADS = os.cpu_count() or 1
JUPYTER_N_RUNS = 15_000
JUPYTER_MAX_STEPS = 300_000
JUPYTER_SEED = 20_260_429


@njit(parallel=True, cache=False, fastmath=False)
def simulate_numba(X0, C, p, lam, n_runs, max_steps, seeds, taus, hits):
    """Simulate independent uncapped first-passage trajectories."""
    for i in prange(n_runs):
        np.random.seed(int(seeds[i]))
        inventory = X0
        tau_i = max_steps
        hit_i = 0
        for t in range(1, max_steps + 1):
            production = 0 if np.random.random() < p else C
            demand = np.random.poisson(lam)
            inventory += production - demand
            if inventory <= 0:
                tau_i = t
                hit_i = 1
                break
        taus[i] = tau_i
        hits[i] = hit_i


def child_seeds(seed: int, count: int) -> np.ndarray:
    sequence = np.random.SeedSequence(seed)
    children = sequence.spawn(count)
    return np.array(
        [child.generate_state(1, dtype=np.uint32)[0] for child in children],
        dtype=np.uint32,
    )


def ig_mle(hit_times: np.ndarray) -> tuple[float, float]:
    tau = np.asarray(hit_times, dtype=np.float64)
    if tau.size == 0 or np.any(tau <= 0):
        raise ValueError("positive observed hit times are required")
    mu_hat = float(tau.mean())
    gap = float(np.mean(1.0 / tau) - 1.0 / mu_hat)
    if gap <= 0:
        raise ValueError("inverse-Gaussian MLE is undefined")
    return mu_hat, 1.0 / gap


def expanded_grid(C: int = 100) -> list[dict]:
    p_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
                0.35, 0.40, 0.45, 0.50]
    utilizations = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                    0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    rows = []
    for p in p_values:
        for utilization in utilizations:
            lam = utilization * C
            delta = (1.0 - p) * C - lam
            if 0.0 < lam < C and 0.5 <= delta <= 20.0:
                rows.append({"p": p, "lam": lam, "delta": delta,
                             "C": C, "source": "expanded"})
    return rows


def fixed_delta_grid(C: int = 100) -> list[dict]:
    p_values = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
                0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    rows = []
    for p in p_values:
        for delta in [2.0, 5.0, 10.0]:
            lam = (1.0 - p) * C - delta
            if 0.0 < lam < C:
                rows.append({"p": p, "lam": lam, "delta": delta,
                             "C": C, "source": "delta_fixed"})
    return rows


def canonical_configs(seed: int) -> list[dict]:
    """Return the 42 fitted points with the original pipeline seed mapping."""
    original = expanded_grid() + fixed_delta_grid()
    config_seeds = child_seeds(seed, len(original))
    unique = []
    seen = set()
    for original_index, (config, config_seed) in enumerate(
        zip(original, config_seeds)
    ):
        key = (round(config["p"], 4), round(config["lam"], 4))
        if key in seen:
            continue
        seen.add(key)
        if config["p"] <= 0.35 and config["delta"] <= 15.0:
            unique.append({
                **config,
                "original_index": original_index,
                "seed_cfg": int(config_seed),
            })
    if len(unique) != 42:
        raise RuntimeError(f"expected 42 canonical points, found {len(unique)}")
    return unique


def beta_model(values, a, b, c):
    p, delta = values
    return a * np.power(p, -(b + c * delta))


def subgroup_stats(values, errors, bins):
    result = {}
    for low, high in bins:
        mask = (values > low) & (values <= high)
        selected = errors[mask]
        result[f"({low},{high}]"] = {
            "n": int(mask.sum()),
            "mean": float(selected.mean()) if selected.size else None,
            "max": float(selected.max()) if selected.size else None,
        }
    return result


def fit_beta(points: pd.DataFrame) -> dict:
    valid = points[
        (points["n_hits"] >= 500) & (points["beta_implied"] > 0.0)
    ].copy()
    if len(valid) < 4:
        raise ValueError(f"only {len(valid)} valid points; at least 4 required")

    p = valid["p"].to_numpy(float)
    delta = valid["delta"].to_numpy(float)
    beta = valid["beta_implied"].to_numpy(float)
    eta_hat = valid["eta_hat"].to_numpy(float)
    lam = valid["lam"].to_numpy(float)
    C = valid["C"].to_numpy(float)
    X0 = valid["X0"].to_numpy(float)

    parameters, covariance = curve_fit(
        beta_model,
        (p, delta),
        beta,
        p0=(0.24, 0.41, 0.011),
        bounds=([1e-4, 1e-4, -1.0], [10.0, 5.0, 1.0]),
        maxfev=20_000,
    )
    a, b, c = map(float, parameters)
    beta_pred = beta_model((p, delta), a, b, c)
    residual_sum = float(np.sum((beta - beta_pred) ** 2))
    total_sum = float(np.sum((beta - beta.mean()) ** 2))
    r_squared = 1.0 - residual_sum / total_sum

    eta_uncorrected = X0 ** 2 / (lam + p * (1.0 - p) * C ** 2)
    eta_corrected = X0 ** 2 / (
        lam + beta_pred * p * (1.0 - p) * C ** 2
    )
    error_unc = 100.0 * np.abs(eta_uncorrected - eta_hat) / eta_hat
    error_corr = 100.0 * np.abs(eta_corrected - eta_hat) / eta_hat

    return {
        "a": a,
        "b": b,
        "c": c,
        "R2_raw_beta": float(r_squared),
        "n_points_used": int(len(valid)),
        "n_points_total": int(len(points)),
        "MAE_uncorrected_eta_pct": float(error_unc.mean()),
        "MAE_corrected_eta_pct": float(error_corr.mean()),
        "max_error_uncorrected_eta_pct": float(error_unc.max()),
        "max_error_corrected_eta_pct": float(error_corr.max()),
        "median_error_uncorrected_eta_pct": float(np.median(error_unc)),
        "median_error_corrected_eta_pct": float(np.median(error_corr)),
        "improvement_factor": float(error_unc.mean() / error_corr.mean()),
        "err_eta_pct_by_p": subgroup_stats(
            p, error_corr, [(0.0, 0.10), (0.10, 0.20), (0.20, 0.35)]
        ),
        "err_eta_pct_by_delta": subgroup_stats(
            delta, error_corr,
            [(0.0, 3.0), (3.0, 7.0), (7.0, 10.0), (10.0, 15.0)],
        ),
        "cov_diag": [float(value) for value in np.diag(covariance)],
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60}m"


def run(args) -> dict:
    output = Path(args.out).resolve()
    checkpoint_csv = output / "checkpoint_points.csv"
    checkpoint_meta = output / "checkpoint_metadata.json"

    signature = {
        "n_runs": args.n_runs,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "metric_denominator": "eta_hat",
        "seed_mode": "SeedSequence.spawn",
    }
    if args.save:
        output.mkdir(parents=True, exist_ok=True)
    if args.save and checkpoint_meta.exists():
        previous = json.loads(checkpoint_meta.read_text(encoding="utf-8"))
        if previous != signature:
            raise RuntimeError(
                "checkpoint settings differ; use another --out directory"
            )
    elif args.save:
        write_json(checkpoint_meta, signature)

    configs = canonical_configs(args.seed)
    if args.save and checkpoint_csv.exists():
        completed = pd.read_csv(checkpoint_csv).to_dict("records")
    else:
        completed = []
    completed_indices = {int(row["original_index"]) for row in completed}

    started = time.perf_counter()
    initial_count = len(completed)
    for position, config in enumerate(configs, start=1):
        if config["original_index"] in completed_indices:
            continue

        run_seeds = child_seeds(config["seed_cfg"], args.n_runs)
        taus = np.empty(args.n_runs, dtype=np.int64)
        hits = np.empty(args.n_runs, dtype=np.uint8)
        point_started = time.perf_counter()
        simulate_numba(
            50, 100, config["p"], config["lam"],
            args.n_runs, args.max_steps, run_seeds, taus, hits,
        )
        observed = taus[hits.astype(bool)]
        n_hits = int(observed.size)
        row = {
            **config,
            "X0": 50,
            "n_runs": args.n_runs,
            "max_steps": args.max_steps,
            "n_hits": n_hits,
            "no_hit_rate": float(1.0 - n_hits / args.n_runs),
        }
        if n_hits:
            mu_hat, eta_hat = ig_mle(observed)
            sigma_implied = 50 ** 2 / eta_hat
            beta_implied = (
                (sigma_implied - config["lam"])
                / (config["p"] * (1.0 - config["p"]) * 100 ** 2)
            )
            row.update({
                "mu_hat": mu_hat,
                "eta_hat": eta_hat,
                "sigma_eff_implied": sigma_implied,
                "beta_implied": beta_implied,
            })
        else:
            row.update({
                "mu_hat": np.nan,
                "eta_hat": np.nan,
                "sigma_eff_implied": np.nan,
                "beta_implied": np.nan,
            })
        completed.append(row)
        completed_indices.add(config["original_index"])
        if args.save:
            pd.DataFrame(completed).sort_values("original_index").to_csv(
                checkpoint_csv, index=False
            )

        finished_now = len(completed) - initial_count
        elapsed = time.perf_counter() - started
        average = elapsed / max(1, finished_now)
        remaining = (len(configs) - len(completed)) * average
        point_time = time.perf_counter() - point_started
        print(
            f"[{len(completed):02d}/{len(configs)}] "
            f"{100*len(completed)/len(configs):5.1f}%  "
            f"p={config['p']:.2f} delta={config['delta']:.1f}  "
            f"hits={n_hits:5d}  point={point_time/60:.1f}m  "
            f"ETA={format_duration(remaining)}",
            flush=True,
        )

    points = pd.DataFrame(completed).sort_values("original_index")
    fit = fit_beta(points)
    summary = {
        "config": signature,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numba_threads": get_num_threads(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "fit": fit,
        "pointwise_results": json.loads(points.to_json(orient="records")),
    }
    if args.save:
        points_file = output / f"beta_points_n{args.n_runs}_T{args.max_steps}.csv"
        fit_file = output / f"beta_fit_n{args.n_runs}_T{args.max_steps}.json"
        summary_file = output / f"summary_n{args.n_runs}_T{args.max_steps}.json"
        points.to_csv(points_file, index=False)
        write_json(fit_file, fit)
        write_json(summary_file, summary)
    return summary


def parse_args():
    if "ipykernel" in sys.modules:
        return argparse.Namespace(
            threads=JUPYTER_THREADS,
            n_runs=JUPYTER_N_RUNS,
            max_steps=JUPYTER_MAX_STEPS,
            seed=JUPYTER_SEED,
            out="beta_final_n15000_T300000",
            save=False,
        )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--n-runs", type=int, default=15_000)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=20_260_429)
    parser.add_argument("--out", default="beta_final_n15000_T300000")
    parser.add_argument(
        "--save",
        action="store_true",
        help="save checkpoints and output files; printing is always enabled",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be at least 1")
    set_num_threads(min(args.threads, os.cpu_count() or args.threads))
    print("Definitive corrected-RNG beta calibration", flush=True)
    print(f"Numba threads: {get_num_threads()}", flush=True)
    print(f"Replications: {args.n_runs:,}; horizon: {args.max_steps:,}", flush=True)
    if args.save:
        print(f"Output/checkpoint: {Path(args.out).resolve()}", flush=True)
        print("The run can be restarted with the same command to resume.", flush=True)
    else:
        print("Output mode: display in notebook only (no files)", flush=True)
    print(flush=True)
    summary = run(args)
    fit = summary["fit"]
    print("\nFINAL RESULTS", flush=True)
    for key in [
        "a", "b", "c", "R2_raw_beta",
        "MAE_uncorrected_eta_pct", "MAE_corrected_eta_pct",
        "max_error_uncorrected_eta_pct", "max_error_corrected_eta_pct",
        "improvement_factor", "n_points_used",
    ]:
        print(f"{key}: {fit[key]}", flush=True)
    print("\nCOMPLETE RESULTS (JSON)", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        main()
    else:
        raise SystemExit(main())
