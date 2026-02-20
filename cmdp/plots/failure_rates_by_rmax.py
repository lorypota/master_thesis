"""
Failure Rates During Training by r_max
========================================

For each r_max, plots per-category failure rates over training steps.
Shows a horizontal line at r_max (the constraint threshold).

Requires dual_history files from training.

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
    help="r_max values to plot (default: all except 1.0)",
)
args = parser.parse_args()

M = args.categories
seeds = list(range(args.seeds[0], args.seeds[1]))
r_max_values = args.r_max_values if args.r_max_values else [v for v in R_MAX_VALUES if v < 1.0]

scenario = get_scenario(M)
active_cats = scenario["active_cats"]
demand_params = scenario["demand_params"]

# Per-period departure rates: 12 * lambda_d for each (cat, period)
cat_period_departures = {}
for cat_idx, cat in enumerate(active_cats):
    cat_period_departures[cat] = {
        0: 12 * demand_params[cat_idx][0][1],  # morning
        1: 12 * demand_params[cat_idx][1][1],  # evening
    }

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


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    return s[1:] if s.startswith("0.") else s


for r_max in r_max_values:
    all_cat_fr = {cat: [] for cat in active_cats}
    all_steps = []

    for seed in seeds:
        fpath = os.path.join(
            RESULTS_DIR, f"dual_history_{M}_cat_{r_max}_{seed}.pkl"
        )
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}, skipping seed {seed}")
            continue

        with open(fpath, "rb") as f:
            history = pickle.load(f)

        # Compute training steps from (repeat, day)
        steps = [entry[0] * 1000 + entry[1] for entry in history]
        all_steps.append(steps)

        # Each entry: (repeat, day, {cat: [morn_f_hat, eve_f_hat]}, avg_base_return)
        # f_hat is per-area per-step failures. Convert to rate by dividing by
        # 12 * lambda_d for each period, then average morning and evening.
        for cat in active_cats:
            fr_pct = []
            for entry in history:
                morn_rate = entry[2][cat][0] / cat_period_departures[cat][0]
                eve_rate = entry[2][cat][1] / cat_period_departures[cat][1]
                fr_pct.append((morn_rate + eve_rate) / 2 * 100)
            all_cat_fr[cat].append(fr_pct)

    if not all_cat_fr[active_cats[0]]:
        print(f"No data for r_max={r_max}, skipping")
        continue

    # Align to shortest seed
    min_len = min(
        *(min(len(s) for s in all_cat_fr[cat]) for cat in active_cats),
        min(len(s) for s in all_steps),
    )
    # Use training steps from first seed as x-axis (all seeds have same structure)
    x_steps = np.array(all_steps[0][:min_len])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for cat in active_cats:
        arr = np.array([s[:min_len] for s in all_cat_fr[cat]])
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)

        ax.plot(
            x_steps,
            means,
            color=CATEGORY_COLORS[cat],
            linewidth=1.5,
            label=CATEGORY_NAMES[cat],
        )
        ax.fill_between(
            x_steps,
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

    ax.set_xlabel("Training step", fontsize=20)
    ax.set_ylabel("Failure rate (%)", fontsize=20)
    ax.tick_params(labelsize=16)

    ax.legend(fontsize=14, loc="best", framealpha=0.9)
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
