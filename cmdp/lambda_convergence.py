import argparse
import os
import pickle

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D  # line style legend entries

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPRESENTATIVE_R_MAX = [0.05, 0.15, 0.35]

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
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

seeds = list(range(args.seeds[0], args.seeds[1]))

GROUPS = [
    ("tight", [0.05, 0.075, 0.10, 0.125, 0.15]),
]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

style_handles = [
    Line2D([0], [0], color="black", linestyle="-", linewidth=1.5, label="Morning"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label="Evening"),
]


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    return s[1:] if s.startswith("0.") else s


def plot_group(r_max_list, group_name, shading=True):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    legend_handles = []

    for r_idx, r_max in enumerate(r_max_list):
        color = colors[r_idx % len(colors)]
        all_morning, all_evening = [], []

        for seed in seeds:
            fpath = os.path.join(
                SCRIPT_DIR,
                f"results/lambda_history_{args.categories}_cat_{r_max}_{seed}.pkl",
            )
            with open(fpath, "rb") as f:
                history = pickle.load(f)

            morning_vals, evening_vals = [], []
            for _repeat, _day, lambdas in history:
                cat_key = list(lambdas.keys())[0]
                morning_vals.append(lambdas[cat_key][0])
                evening_vals.append(lambdas[cat_key][1])

            all_morning.append(morning_vals)
            all_evening.append(evening_vals)

        min_len = min(len(m) for m in all_morning)
        morning_arr = np.array([m[:min_len] for m in all_morning])
        evening_arr = np.array([e[:min_len] for e in all_evening])
        steps = np.arange(min_len)

        morning_mean = np.mean(morning_arr, axis=0)
        morning_std = np.std(morning_arr, axis=0)
        evening_mean = np.mean(evening_arr, axis=0)
        evening_std = np.std(evening_arr, axis=0)

        ax.plot(steps, morning_mean, color=color, linestyle="-", linewidth=1.5)
        if shading:
            ax.fill_between(
                steps,
                morning_mean - 1.96 * morning_std,
                morning_mean + 1.96 * morning_std,
                color=color,
                alpha=0.15,
            )

        ax.plot(steps, evening_mean, color=color, linestyle="--", linewidth=1.5)
        if shading:
            ax.fill_between(
                steps,
                evening_mean - 1.96 * evening_std,
                evening_mean + 1.96 * evening_std,
                color=color,
                alpha=0.08,
            )

        legend_handles.append(
            mpatches.Patch(color=color, label=rf"$r_{{max}}$={_fmt(r_max)}")
        )

    ax.set_xlabel("Dual update step", fontsize=26)
    ax.set_ylabel(r"$\lambda$", fontsize=26)
    ax.tick_params(labelsize=22)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
    ax.legend(
        handles=legend_handles + style_handles, fontsize=18, loc="best", framealpha=0.4
    )
    plt.tight_layout()

    if args.save:
        plt.savefig(
            os.path.join(
                SCRIPT_DIR,
                f"plots/lambda_convergence_{args.categories}_cat_{group_name}.png",
            ),
            format="png",
        )
    plt.show()


plot_group(GROUPS[0][1], GROUPS[0][0], shading=True)
