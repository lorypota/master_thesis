import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import R_MAX_VALUES, compute_failure_thresholds
from common.config import get_scenario

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
parser.add_argument(
    "--all-categories",
    action="store_true",
    help="Show all categories in subplots",
)
args = parser.parse_args()

scenario = get_scenario(args.categories)
demand_params = scenario["demand_params"]
active_cats = scenario["active_cats"]

# Shape: (num_r_max, num_seeds, num_active_cats, 2) where last dim = [morning, evening]
data = np.load(
    os.path.join(
        SCRIPT_DIR,
        f"results/failure_rates_per_cat_period_{args.categories}_cat_10seeds.npy",
    )
)

num_r_max = data.shape[0]
num_active_cats = data.shape[2]
r_max_labels = [str(r) for r in R_MAX_VALUES[:num_r_max]]

sns.set(style="whitegrid")

if args.all_categories:
    fig, axes = plt.subplots(
        1, num_active_cats, figsize=(6 * num_active_cats, 6), dpi=100
    )
    if num_active_cats == 1:
        axes = [axes]

    for cat_idx in range(num_active_cats):
        ax = axes[cat_idx]
        morning_means = np.mean(data[:, :, cat_idx, 0], axis=1)
        evening_means = np.mean(data[:, :, cat_idx, 1], axis=1)
        morning_stds = np.std(data[:, :, cat_idx, 0], axis=1)
        evening_stds = np.std(data[:, :, cat_idx, 1], axis=1)

        x = np.arange(num_r_max)
        width = 0.35

        bars_m = ax.bar(
            x - width / 2,
            morning_means,
            width,
            yerr=morning_stds,
            label="Morning",
            color=sns.color_palette("viridis", 11)[2],
            alpha=0.8,
            capsize=3,
        )
        bars_e = ax.bar(
            x + width / 2,
            evening_means,
            width,
            yerr=evening_stds,
            label="Evening",
            color=sns.color_palette("viridis", 11)[7],
            alpha=0.8,
            capsize=3,
        )

        # Overlay r_max threshold lines for constrained category (cat 0)
        if cat_idx == 0:
            constrained_cat = active_cats[0]
            for r_idx, r_max in enumerate(R_MAX_VALUES[:num_r_max]):
                thresholds = compute_failure_thresholds(
                    r_max, demand_params, active_cats, {constrained_cat}
                )
                threshold_m = thresholds[constrained_cat][0]
                threshold_e = thresholds[constrained_cat][1]
                ax.plot(
                    [r_idx - 0.4, r_idx + 0.4],
                    [threshold_m, threshold_m],
                    color="red",
                    linewidth=1.5,
                    linestyle="--",
                )
                ax.plot(
                    [r_idx - 0.4, r_idx + 0.4],
                    [threshold_e, threshold_e],
                    color="darkred",
                    linewidth=1.5,
                    linestyle=":",
                )

        ax.set_xlabel(r"$r_{max}$", fontsize=18)
        ax.set_ylabel("Avg. failures per station", fontsize=18)
        ax.set_title(f"Category {cat_idx}", fontsize=20)
        ax.set_xticks(x)
        ax.set_xticklabels(r_max_labels, fontsize=14)
        ax.tick_params(labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(
            True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7
        )

    plt.tight_layout()
    if args.save:
        plt.savefig(
            os.path.join(
                SCRIPT_DIR,
                f"plots/failure_rates_all_cats_{args.categories}_cat.png",
            ),
            format="png",
        )
    plt.show()

else:
    # Single plot for constrained category (cat 0)
    cat_idx = 0
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    morning_means = np.mean(data[:, :, cat_idx, 0], axis=1)
    evening_means = np.mean(data[:, :, cat_idx, 1], axis=1)
    morning_stds = np.std(data[:, :, cat_idx, 0], axis=1)
    evening_stds = np.std(data[:, :, cat_idx, 1], axis=1)

    x = np.arange(num_r_max)
    width = 0.35

    ax.bar(
        x - width / 2,
        morning_means,
        width,
        yerr=morning_stds,
        label="Morning",
        color=sns.color_palette("viridis", 11)[2],
        alpha=0.8,
        capsize=3,
    )
    ax.bar(
        x + width / 2,
        evening_means,
        width,
        yerr=evening_stds,
        label="Evening",
        color=sns.color_palette("viridis", 11)[7],
        alpha=0.8,
        capsize=3,
    )

    # Overlay r_max constraint thresholds
    constrained_cat = active_cats[0]
    for r_idx, r_max in enumerate(R_MAX_VALUES[:num_r_max]):
        thresholds = compute_failure_thresholds(
            r_max, demand_params, active_cats, {constrained_cat}
        )
        threshold_m = thresholds[constrained_cat][0]
        threshold_e = thresholds[constrained_cat][1]
        lbl_m = "Morning threshold" if r_idx == 0 else None
        lbl_e = "Evening threshold" if r_idx == 0 else None
        ax.plot(
            [r_idx - 0.4, r_idx + 0.4],
            [threshold_m, threshold_m],
            color="red",
            linewidth=2,
            linestyle="--",
            label=lbl_m,
        )
        ax.plot(
            [r_idx - 0.4, r_idx + 0.4],
            [threshold_e, threshold_e],
            color="darkred",
            linewidth=2,
            linestyle=":",
            label=lbl_e,
        )

    ax.set_xlabel(r"$r_{max}$", fontsize=36)
    ax.set_ylabel("Avg. failures per station", fontsize=28)
    ax.set_xticks(x)
    ax.set_xticklabels(r_max_labels, fontsize=34)
    ax.tick_params(labelsize=28)
    ax.legend(fontsize=18, loc="best", framealpha=0.4)
    ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)

    plt.tight_layout()
    if args.save:
        plt.savefig(
            os.path.join(
                SCRIPT_DIR,
                f"plots/failure_rates_cat0_{args.categories}_cat.png",
            ),
            format="png",
        )
    plt.show()
