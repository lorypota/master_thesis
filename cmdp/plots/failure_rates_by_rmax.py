"""
Failure Rates During Training by r_max
========================================

For each r_max, plots per-category failure rates and base costs
over dual update steps. Shows a horizontal line at r_max (the constraint).

Requires dual_history files from training (saved with updated training.py).

Usage:
    uv run cmdp/plots/failure_rates_by_rmax.py --categories 5
    uv run cmdp/plots/failure_rates_by_rmax.py --categories 5 --save
    uv run cmdp/plots/failure_rates_by_rmax.py --categories 5 --r-max-values 0.05 0.075 0.1
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import R_MAX_VALUES
from common.config import get_scenario

PLOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PLOT_DIR, "..", "results")

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument(
    "--seeds",
    nargs=2,
    type=int,
    default=[100, 110],
    help="Seed range [start, end)",
)
parser.add_argument(
    "--r-max-values",
    nargs="+",
    type=float,
    default=None,
    help="r_max values to plot (default: all)",
)
args = parser.parse_args()

M = args.categories
seeds = list(range(args.seeds[0], args.seeds[1]))
r_max_values = args.r_max_values if args.r_max_values else R_MAX_VALUES

scenario = get_scenario(M)
active_cats = scenario["active_cats"]
demand_params = scenario["demand_params"]

# Departure rates per category (sum morning + evening lambda_d * 12h each)
cat_departures_per_day = {}
for cat_idx, cat in enumerate(active_cats):
    lambda_d_morn = demand_params[cat_idx][0][1]
    lambda_d_eve = demand_params[cat_idx][1][1]
    cat_departures_per_day[cat] = 12 * lambda_d_morn + 12 * lambda_d_eve

CATEGORY_NAMES = {
    0: "Remote (cat 0)",
    1: "Suburban-remote (cat 1)",
    2: "Suburban (cat 2)",
    3: "Suburban-central (cat 3)",
    4: "Central (cat 4)",
}
CATEGORY_COLORS = {
    0: "#2ca02c",
    1: "#8c564b",
    2: "#ff7f0e",
    3: "#9467bd",
    4: "#1f77b4",
}
CATEGORY_MARKERS = {0: "o", 1: "^", 2: "D", 3: "v", 4: "s"}


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    return s[1:] if s.startswith("0.") else s


for r_max in r_max_values:
    # Load dual_history for all seeds
    all_cat_fr = {cat: [] for cat in active_cats}
    all_base_costs = []

    for seed in seeds:
        fpath = os.path.join(
            RESULTS_DIR, f"dual_history_{M}_cat_{r_max}_{seed}.pkl"
        )
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}, skipping seed {seed}")
            continue

        with open(fpath, "rb") as f:
            history = pickle.load(f)

        # Each entry: (repeat, day, {cat: [morn_f_hat, eve_f_hat]}, avg_base_return)
        for cat in active_cats:
            # Convert absolute failures to failure rate %
            fr_pct = [
                (entry[2][cat][0] + entry[2][cat][1])
                / cat_departures_per_day[cat]
                * 100
                for entry in history
            ]
            all_cat_fr[cat].append(fr_pct)

        all_base_costs.append([-entry[3] for entry in history])

    if not all_base_costs:
        print(f"No data for r_max={r_max}, skipping")
        continue

    # Align to shortest seed
    min_len = min(
        min(len(s) for s in all_base_costs),
        *(min(len(s) for s in all_cat_fr[cat]) for cat in active_cats),
    )
    steps = np.arange(min_len)

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for cat in active_cats:
        arr = np.array([s[:min_len] for s in all_cat_fr[cat]])
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)

        ax.plot(
            steps,
            means,
            color=CATEGORY_COLORS[cat],
            linewidth=1.5,
            label=CATEGORY_NAMES[cat],
        )
        ax.fill_between(
            steps,
            means - stds,
            means + stds,
            color=CATEGORY_COLORS[cat],
            alpha=0.15,
        )

    # Horizontal line at r_max threshold
    ax.axhline(
        y=r_max * 100,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=rf"$r_{{max}}$ = {_fmt(r_max)} ({r_max*100:g}%)",
    )

    ax.set_xlabel("Dual update step", fontsize=20)
    ax.set_ylabel("Failure rate (%)", fontsize=20)
    ax.tick_params(labelsize=16)

    # Secondary y-axis for base costs
    ax2 = ax.twinx()
    cost_arr = np.array([s[:min_len] for s in all_base_costs])
    cost_means = np.mean(cost_arr, axis=0)
    cost_stds = np.std(cost_arr, axis=0)

    ax2.plot(
        steps,
        cost_means,
        "--",
        color="#d62728",
        linewidth=1.5,
        alpha=0.8,
        label="Base costs",
    )
    ax2.fill_between(
        steps,
        cost_means - cost_stds,
        cost_means + cost_stds,
        color="#d62728",
        alpha=0.1,
    )
    ax2.set_ylabel(r"Costs without $\lambda$ penalty", fontsize=20, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=16)
    ax2.grid(False)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=14,
        loc="best",
        framealpha=0.9,
    )

    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.5)
    ax.set_title(rf"$r_{{max}}$ = {_fmt(r_max)}", fontsize=22)
    plt.tight_layout()

    if args.save:
        out_path = os.path.join(
            PLOT_DIR, f"failure_rates_rmax{_fmt(r_max)}_{M}_cat.png"
        )
        plt.savefig(out_path, format="png", bbox_inches="tight", dpi=150)
        print(f"Saved: {out_path}")

    plt.show()
