import argparse
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D  # line style legend entries

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
R_MAX_VALUES = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
REPRESENTATIVE_R_MAX = [0.05, 0.15, 0.35]

parser = argparse.ArgumentParser()
parser.add_argument("--cat", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument(
    "--r-max-values",
    nargs="+",
    type=float,
    default=None,
    help="r_max values to plot (default: 0.05, 0.15, 0.35)",
)
parser.add_argument(
    "--seeds",
    nargs=2,
    type=int,
    default=[100, 110],
    help="Seed range [start, end)",
)
args = parser.parse_args()

r_max_to_plot = args.r_max_values if args.r_max_values else REPRESENTATIVE_R_MAX
seeds = list(range(args.seeds[0], args.seeds[1]))
num_seeds = len(seeds)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

legend_handles = []

for r_idx, r_max in enumerate(r_max_to_plot):
    color = colors[r_idx % len(colors)]

    # Collect lambda histories across seeds
    all_morning = []
    all_evening = []

    for seed in seeds:
        fpath = os.path.join(
            SCRIPT_DIR,
            f"results/lambda_history_{args.cat}_cat_{r_max}_{seed}.pkl",
        )
        with open(fpath, "rb") as f:
            history = pickle.load(f)

        # Extract constrained category (cat 0) lambda values
        # Each entry: (repeat, day, {cat: [lambda_morning, lambda_evening]})
        morning_vals = []
        evening_vals = []
        for _repeat, _day, lambdas in history:
            # Use first constrained category (cat 0)
            cat_key = list(lambdas.keys())[0]
            morning_vals.append(lambdas[cat_key][0])
            evening_vals.append(lambdas[cat_key][1])

        all_morning.append(morning_vals)
        all_evening.append(evening_vals)

    # Align to shortest length across seeds
    min_len = min(len(m) for m in all_morning)
    morning_arr = np.array([m[:min_len] for m in all_morning])
    evening_arr = np.array([e[:min_len] for e in all_evening])

    steps = np.arange(min_len)

    # Mean and 95% CI
    morning_mean = np.mean(morning_arr, axis=0)
    morning_std = np.std(morning_arr, axis=0)
    evening_mean = np.mean(evening_arr, axis=0)
    evening_std = np.std(evening_arr, axis=0)

    ax.plot(
        steps,
        morning_mean,
        color=color,
        linestyle="-",
        linewidth=1.5,
        label=rf"$r_{{max}}$={r_max} morning",
    )
    ax.fill_between(
        steps,
        morning_mean - 1.96 * morning_std,
        morning_mean + 1.96 * morning_std,
        color=color,
        alpha=0.15,
    )

    ax.plot(
        steps,
        evening_mean,
        color=color,
        linestyle="--",
        linewidth=1.5,
        label=rf"$r_{{max}}$={r_max} evening",
    )
    ax.fill_between(
        steps,
        evening_mean - 1.96 * evening_std,
        evening_mean + 1.96 * evening_std,
        color=color,
        alpha=0.08,
    )

    # Legend patches
    legend_handles.append(mpatches.Patch(color=color, label=rf"$r_{{max}}$={r_max}"))

ax.set_xlabel("Dual update step", fontsize=26)
ax.set_ylabel(r"$\lambda$", fontsize=26)
ax.tick_params(labelsize=22)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)

style_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Morning"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Evening"),
]
ax.legend(
    handles=legend_handles + style_handles, fontsize=18, loc="best", framealpha=0.4
)

plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(SCRIPT_DIR, f"plots/lambda_convergence_{args.cat}_cat.png"),
        format="png",
    )
plt.show()
