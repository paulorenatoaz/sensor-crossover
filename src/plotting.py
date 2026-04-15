# ---------------------------------------------------------------------------
# plotting.py — Publication-quality figures
# ---------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as cfg


def set_publication_style():
    """Configure matplotlib for paper-quality figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (4.5, 3.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
    })


def plot_crossover(
    summary_d1: pd.DataFrame,
    summary_d2: pd.DataFrame,
    scenario_name: str,
    n_stars: list[float] | None = None,
    save_path: str | None = None,
):
    """Plot error vs n for d=1 and d=2 with error bands.

    Parameters
    ----------
    summary_d1, summary_d2 : DataFrames from summarize_results()
        Must have columns [n, mean_error, std_error].
    n_stars : list of crossover points to draw as vertical lines.
    """
    set_publication_style()
    fig, ax = plt.subplots()

    for summary, d_label, color, marker in [
        (summary_d1, "$d=1$ sensor",  "#1f77b4", "o"),
        (summary_d2, "$d=2$ sensors", "#d62728", "s"),
    ]:
        n = summary["n"].values
        mean = summary["mean_error"].values
        std = summary["std_error"].values

        ax.plot(n, mean, marker=marker, color=color, label=d_label)
        ax.fill_between(n, mean - std, mean + std, alpha=0.15, color=color)

    if n_stars:
        vline_colors = ["gray", "#9333ea"]  # second line in purple for visibility
        for i, ns in enumerate(n_stars):
            c = vline_colors[i % len(vline_colors)]
            ax.axvline(ns, color=c, linestyle="--", linewidth=1, alpha=0.7)
            # Stagger label positions vertically so they don't overlap
            y_frac = 0.95 - i * 0.08
            ax.text(
                ns * 1.08, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac,
                f"$n^*\\approx{ns:.0f}$",
                fontsize=9, color=c,
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training samples $n$")
    ax.set_ylabel("Classification error")
    ax.set_title(f"Crossover — {scenario_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        print(f"Saved {save_path}")
    plt.close(fig)


def plot_scenario_comparison(
    all_summaries: dict,
    save_path: str | None = None,
):
    """Plot Δ error = error(d=1) − error(d=2) vs n for all scenarios.

    Parameters
    ----------
    all_summaries : dict mapping scenario → {'d1': summary_df, 'd2': summary_df}
    """
    set_publication_style()
    fig, ax = plt.subplots()

    colors = {"high-correlation": "#2ca02c", "mid-correlation": "#1f77b4", "low-correlation": "#d62728"}
    markers = {"high-correlation": "^", "mid-correlation": "o", "low-correlation": "v"}

    for scenario, data in all_summaries.items():
        n = data["d1"]["n"].values
        delta = data["d1"]["mean_error"].values - data["d2"]["mean_error"].values
        ax.plot(
            n, delta,
            marker=markers.get(scenario, "o"),
            color=colors.get(scenario, "black"),
            label=scenario.capitalize(),
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training samples $n$")
    ax.set_ylabel("$\\Delta$ error ($d{=}1$ minus $d{=}2$)")
    ax.set_title("Benefit of adding a second sensor")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        print(f"Saved {save_path}")
    plt.close(fig)


def plot_correlation_vs_crossover(
    corr_values: list[float],
    n_star_values: list[float | None],
    labels: list[str],
    save_path: str | None = None,
):
    """Plot estimated n* vs correlation with R for different B sensors."""
    set_publication_style()
    fig, ax = plt.subplots()

    valid = [(c, n, l) for c, n, l in zip(corr_values, n_star_values, labels) if n is not None]
    if not valid:
        print("No crossover points found — skipping correlation plot")
        plt.close(fig)
        return

    cs, ns, ls = zip(*valid)
    ax.scatter(cs, ns, s=60, zorder=3)
    for c, n, l in valid:
        ax.annotate(l, (c, n), textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel("corr($B$, $R$)")
    ax.set_ylabel("Crossover point $n^*$")
    ax.set_title("Correlation vs crossover point")
    ax.grid(True, alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        print(f"Saved {save_path}")
    plt.close(fig)


def plot_delta_with_ci(
    delta_stats: dict,
    n_stars: dict | None = None,
    save_path: str | None = None,
):
    """Plot Δ(n) = E(d=1)−E(d=2) with 95% CI bands and significance markers.

    Parameters
    ----------
    delta_stats : dict mapping scenario → DataFrame with
        [n, mean_delta, ci_lo, ci_hi, p_value].
    n_stars : dict mapping scenario → list[float] crossover points.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    colors = {
        "high-correlation": "#2ca02c",
        "mid-correlation": "#1f77b4",
        "low-correlation": "#d62728",
    }
    markers = {"high-correlation": "^", "mid-correlation": "o", "low-correlation": "v"}

    for scenario, ds in delta_stats.items():
        c = colors.get(scenario, "black")
        m = markers.get(scenario, "o")
        n = ds["n"].values
        mean = ds["mean_delta"].values
        lo = ds["ci_lo"].values
        hi = ds["ci_hi"].values
        sig = ds["p_value"].values < 0.05

        ax.plot(n, mean, marker=m, color=c, label=scenario, zorder=3)
        ax.fill_between(n, lo, hi, alpha=0.15, color=c)

        # Ring for non-significant points (CI includes zero)
        ns_idx = ~sig
        if ns_idx.any():
            ax.scatter(
                n[ns_idx], mean[ns_idx],
                marker="o", s=80, facecolors="none", edgecolors=c,
                linewidths=1.5, zorder=4, label=None,
            )

    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Training samples $n$")
    ax.set_ylabel("$\\Delta$ error ($d{=}1$ minus $d{=}2$)")
    ax.set_title("Paired error difference with 95% CI")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Annotation
    ylim = ax.get_ylim()
    span = ylim[1] - ylim[0]
    ax.text(
        2.2, ylim[1] - 0.05 * span, "$d{=}2$ better $\\uparrow$",
        fontsize=7, color="gray", ha="left", va="top",
    )
    ax.text(
        2.2, ylim[0] + 0.05 * span, "$d{=}1$ better $\\downarrow$",
        fontsize=7, color="gray", ha="left", va="bottom",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
        fig.savefig(save_path.replace(".pdf", ".png"))
        print(f"Saved {save_path}")
    plt.close(fig)
