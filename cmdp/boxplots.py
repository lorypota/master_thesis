import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import R_MAX_VALUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

gini = np.load(
    os.path.join(SCRIPT_DIR, f"results/gini_{args.categories}_cat_10seeds.npy")
).transpose()
cost_reb = np.load(
    os.path.join(SCRIPT_DIR, f"results/cost_reb_{args.categories}_cat_10seeds.npy")
).transpose()
cost_fail = np.load(
    os.path.join(SCRIPT_DIR, f"results/cost_fail_{args.categories}_cat_10seeds.npy")
).transpose()
cost_bikes = np.load(
    os.path.join(SCRIPT_DIR, f"results/cost_bikes_{args.categories}_cat_10seeds.npy")
).transpose()

r_max_labels = [str(r) for r in R_MAX_VALUES]
num_r_max = len(R_MAX_VALUES)

# GINI INDEX

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[7]
box = ax.boxplot(gini, patch_artist=True, notch=False, vert=True, widths=0.6)
for patch in box["boxes"]:
    patch.set_facecolor(box_color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.8)
    patch.set_linewidth(1.5)
for whisker in box["whiskers"]:
    whisker.set(color="black", linewidth=1.5, linestyle="--")
for cap in box["caps"]:
    cap.set(color="black", linewidth=1.5)
for median in box["medians"]:
    median.set(color="black", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Gini index", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.tick_params(labelsize=34)
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(SCRIPT_DIR, f"plots/boxplot_gini_{args.categories}_cat.png"),
        format="png",
    )
plt.show()

# REBALANCING COSTS

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[2]
box = ax.boxplot(cost_reb, patch_artist=True, notch=False, vert=True, widths=0.6)
for patch in box["boxes"]:
    patch.set_facecolor(box_color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.8)
    patch.set_linewidth(1.5)
for whisker in box["whiskers"]:
    whisker.set(color="black", linewidth=1.5, linestyle="--")
for cap in box["caps"]:
    cap.set(color="black", linewidth=1.5)
for median in box["medians"]:
    median.set(color="gold", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Weighted reb. op's", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.tick_params(labelsize=34)
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(SCRIPT_DIR, f"plots/boxplot_costs_reb_{args.categories}_cat.png"),
        format="png",
    )
plt.show()

# FAILURE COSTS

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[2]
box = ax.boxplot(cost_fail, patch_artist=True, notch=False, vert=True, widths=0.6)
for patch in box["boxes"]:
    patch.set_facecolor(box_color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.8)
    patch.set_linewidth(1.5)
for whisker in box["whiskers"]:
    whisker.set(color="black", linewidth=1.5, linestyle="--")
for cap in box["caps"]:
    cap.set(color="black", linewidth=1.5)
for median in box["medians"]:
    median.set(color="gold", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Failure rate [%]", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.tick_params(labelsize=34)
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            SCRIPT_DIR, f"plots/boxplot_costs_fails_{args.categories}_cat.png"
        ),
        format="png",
    )
plt.show()

# NUMBER OF VEHICLES

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
box_color = sns.color_palette("viridis", 11)[2]
box = ax.boxplot(cost_bikes, patch_artist=True, notch=False, vert=True, widths=0.6)
for patch in box["boxes"]:
    patch.set_facecolor(box_color)
    patch.set_edgecolor("black")
    patch.set_alpha(0.8)
    patch.set_linewidth(1.5)
for whisker in box["whiskers"]:
    whisker.set(color="black", linewidth=1.5, linestyle="--")
for cap in box["caps"]:
    cap.set(color="black", linewidth=1.5)
for median in box["medians"]:
    median.set(color="gold", linewidth=1.5)
for flier in box["fliers"]:
    flier.set(marker="o", color="red", alpha=0.75)
ax.set_xlabel(r"$r_{max}$", fontsize=36)
ax.set_ylabel("Number of vehicles", fontsize=36)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
ax.set_xticks(range(1, num_r_max + 1))
ax.set_xticklabels(r_max_labels, fontsize=34)
ax.tick_params(labelsize=34)
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(
            SCRIPT_DIR, f"plots/boxplot_costs_bikes_{args.categories}_cat.png"
        ),
        format="png",
    )
plt.show()
