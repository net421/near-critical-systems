"""Phase 4 plotting helpers. Uses Agg backend and writes PDF files."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


def plot_cost_ratio(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path).sort_values("kp_over_kf")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["kp_over_kf"], df["ratio_opt_to_RTF"], "o-",
            label="g* / g_RTF")
    ax.plot(df["kp_over_kf"], df["ratio_opt_to_AB"], "s--",
            label="g* / g_AB_best")
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("K_p / K_f")
    ax.set_ylabel("Cost ratio")
    ax.set_title("Threshold policy cost ratios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, out_path)


def plot_heatmap_sstar(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df = df[df["valid"].astype(bool)]

    fig, ax = plt.subplots(figsize=(6, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No valid rows", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_path)

    pivot = df.pivot(index="delta", columns="p", values="s_star")
    pivot = pivot.sort_index()
    pivot = pivot.reindex(sorted(pivot.columns), axis=1)

    im = ax.imshow(pivot.values, origin="lower", aspect="auto",
                   cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{x:.1f}" for x in pivot.index])
    ax.set_xlabel("p")
    ax.set_ylabel("delta")
    ax.set_title("Optimal threshold s*")
    fig.colorbar(im, ax=ax, label="s*")
    return _save(fig, out_path)


def plot_smax(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df = df[df["valid"].astype(bool)].sort_values("S_max")

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(df["S_max"], df["s_star"], "o-", color="C0")
    ax1.set_xlabel("S_max")
    ax1.set_ylabel("s*", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["S_max"], df["ratio_star_to_RTF"], "s--", color="C3")
    ax2.set_ylabel("g* / g_RTF", color="C3")
    ax2.tick_params(axis="y", labelcolor="C3")
    ax1.set_title("S_max sweep")

    return _save(fig, out_path)


def plot_hazard(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df = df[df["valid"].astype(bool)]

    fig, ax = plt.subplots(figsize=(6, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No valid rows", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_path)

    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["h_initial"], w, label="initial")
    ax.bar(x + w / 2, df["h_final"], w, label="final")
    ax.set_xticks(x)
    ax.set_xticklabels(df["config"], rotation=30, ha="right")
    ax.set_ylabel("hazard")
    ax.set_title("Capped deterministic hazard diagnostics")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return _save(fig, out_path)


def plot_robustness(csv_path: Path, out_path: Path) -> Path:
    df = pd.read_csv(csv_path)
    df = df[df["valid"].astype(bool)]

    fig, ax = plt.subplots(figsize=(7, 4))
    if df.empty:
        ax.text(0.5, 0.5, "No valid rows", ha="center", va="center")
        ax.set_axis_off()
        return _save(fig, out_path)

    x = np.arange(len(df))
    ax.bar(x, df["ratio_star_to_RTF"])
    ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(df["scenario"], rotation=30, ha="right")
    ax.set_ylabel("g* / g_RTF")
    ax.set_title("Robustness scenarios")
    ax.grid(True, axis="y", alpha=0.3)

    return _save(fig, out_path)


def make_all_plots(paper_dir: Path, fig_dir: Path) -> list[Path]:
    """Generate all paper plots from CSVs."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        ("exp_cost_ratio.csv", "fig_cost_ratio.pdf", plot_cost_ratio),
        ("exp_heatmap.csv", "fig_heatmap_sstar.pdf", plot_heatmap_sstar),
        ("exp_smax.csv", "fig_smax_sweep.pdf", plot_smax),
        ("diagnostic_hazard.csv", "fig_hazard.pdf", plot_hazard),
        ("exp_robustness.csv", "fig_robustness.pdf", plot_robustness),
    ]

    out: list[Path] = []
    for csv_name, fig_name, fn in jobs:
        csv_path = paper_dir / csv_name
        if csv_path.exists():
            try:
                out.append(fn(csv_path, fig_dir / fig_name))
            except Exception as exc:
                print(f"[plots] WARNING: failed {csv_name}: {exc}")
    return out
