"""
Failure Rates During Training by r_max
========================================

For each r_max, plots per-category failure rates over training steps
(left y-axis) and rebalancing+deviation costs (right y-axis).
Shows a horizontal line at r_max (the constraint threshold).

Rebalancing costs are derived from dual_history:
  base_return = -(failures + reb_dev_cost)
  => reb_dev_cost = -base_return - failures

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
r_max_values = args.r_max_values if args.r_max_values else [v for v in R_MAX_VALUES]

scenario = get_scenario(M)
active_cats = scenario["active_cats"]
demand_params = scenario["demand_params"]
node_list = scenario["node_list"]

# Per-period departure rates: 12 * lambda_d for each (cat, period)
cat_period_departures = {}
for cat_idx, cat in enumerate(active_cats):
    cat_period_departures[cat] = {
        0: 12 * demand_params[cat_idx][0][1],  # morning
        1: 12 * demand_params[cat_idx][1][1],  # evening
    }

CATEGORY_NAMES = {
    0: "Cat 0 (remote)",
    1: "Cat 1",
    2: "Cat 2",
    3: "Cat 3",
    4: "Cat 4 (central)",
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
    all_cat_fr_morn = {cat: [] for cat in active_cats}
    all_cat_fr_eve = {cat: [] for cat in active_cats}
    all_reb_costs = []
    all_steps = []

    for seed in seeds:
        fpath = os.path.join(RESULTS_DIR, f"dual_history_{M}_cat_{r_max}_{seed}.pkl")
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
        # 12 * lambda_d for each period.
        reb_costs_seed = []
        for cat in active_cats:
            morn_pct = []
            eve_pct = []
            for entry in history:
                morn_pct.append(entry[2][cat][0] / cat_period_departures[cat][0] * 100)
                eve_pct.append(entry[2][cat][1] / cat_period_departures[cat][1] * 100)
            all_cat_fr_morn[cat].append(morn_pct)
            all_cat_fr_eve[cat].append(eve_pct)

        # Use direct reb_costs (5th element) if available, else derive from base_return
        for entry in history:
            if len(entry) >= 5:
                reb_costs_seed.append(entry[4])
            else:
                avg_base_return = entry[3]
                total_failures = 0.0
                for cat_idx, cat in enumerate(active_cats):
                    for p in (0, 1):
                        total_failures += entry[2][cat][p] * node_list[cat_idx]
                reb_costs_seed.append(-avg_base_return - total_failures)
        all_reb_costs.append(reb_costs_seed)

    if not all_cat_fr_morn[active_cats[0]]:
        print(f"No data for r_max={r_max}, skipping")
        continue

    # Align to shortest seed
    min_len = min(
        *(min(len(s) for s in all_cat_fr_morn[cat]) for cat in active_cats),
        *(min(len(s) for s in all_cat_fr_eve[cat]) for cat in active_cats),
        min(len(s) for s in all_steps),
        min(len(s) for s in all_reb_costs),
    )
    x_steps = np.array(all_steps[0][:min_len])

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

    for cat in active_cats:
        morn_arr = np.array([s[:min_len] for s in all_cat_fr_morn[cat]])
        eve_arr = np.array([s[:min_len] for s in all_cat_fr_eve[cat]])
        morn_means = np.mean(morn_arr, axis=0)
        morn_stds = np.std(morn_arr, axis=0)
        eve_means = np.mean(eve_arr, axis=0)
        eve_stds = np.std(eve_arr, axis=0)

        # Morning: solid line (with label)
        ax.plot(
            x_steps,
            morn_means,
            color=CATEGORY_COLORS[cat],
            linewidth=1.5,
            label=CATEGORY_NAMES[cat],
        )
        ax.fill_between(
            x_steps,
            morn_means - morn_stds,
            morn_means + morn_stds,
            color=CATEGORY_COLORS[cat],
            alpha=0.15,
        )

        # Evening: dotted line (no label to avoid legend duplication)
        ax.plot(
            x_steps,
            eve_means,
            color=CATEGORY_COLORS[cat],
            linewidth=1.5,
            linestyle=":",
        )
        ax.fill_between(
            x_steps,
            eve_means - eve_stds,
            eve_means + eve_stds,
            color=CATEGORY_COLORS[cat],
            alpha=0.1,
        )

    # Horizontal line at r_max threshold
    ax.axhline(
        y=r_max * 100,
        color="black",
        linestyle="--",
        linewidth=1.5,
        alpha=0.6,
        label=rf"$r_{{max}}$ {r_max * 100:g}%",
    )

    ax.set_xlabel("Training step", fontsize=20)
    ax.set_ylabel("Failure rate (%)", fontsize=20)
    ax.tick_params(labelsize=16)

    # Secondary y-axis for rebalancing + deviation costs
    ax2 = ax.twinx()
    cost_arr = np.array([s[:min_len] for s in all_reb_costs])
    cost_means = np.mean(cost_arr, axis=0)
    cost_stds = np.std(cost_arr, axis=0)

    ax2.plot(
        x_steps,
        cost_means,
        color="#d62728",
        linewidth=1.5,
        linestyle="-.",
        alpha=0.8,
        label="Costs",
    )
    ax2.fill_between(
        x_steps,
        cost_means - cost_stds,
        cost_means + cost_stds,
        color="#d62728",
        alpha=0.1,
    )
    ax2.set_ylabel("Costs (reb. + dev.)", fontsize=20, color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728", labelsize=16)
    ax2.grid(False)

    # Combine legends (compact, below title)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2,
        labels1 + labels2,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        framealpha=0.9,
        handlelength=1.5,
        handletextpad=0.4,
        columnspacing=0.8,
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
