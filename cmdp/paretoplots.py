import argparse
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import R_MAX_VALUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("--categories", default=5, type=int)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()
cat = args.categories

gini = np.load(
    os.path.join(SCRIPT_DIR, f"results/gini_{cat}_cat_10seeds.npy")
).transpose()
cost = np.load(
    os.path.join(SCRIPT_DIR, f"results/cost_{cat}_cat_10seeds.npy")
).transpose()
constraint_sat = np.load(
    os.path.join(SCRIPT_DIR, f"results/constraint_sat_{cat}_cat_10seeds.npy")
).transpose()

num_r_max = len(R_MAX_VALUES)
avg_ginis = [np.mean(gini[:, i]) for i in range(num_r_max)]
avg_costs = [np.mean(cost[:, i]) for i in range(num_r_max)]
# Fraction of seeds satisfying constraints per r_max
sat_fractions = [np.mean(constraint_sat[:, i]) for i in range(num_r_max)]


def compute_pareto_frontier(costs, ginis):
    """Return indices of Pareto-optimal points (minimizing both cost and gini)."""
    n = len(costs)
    is_pareto = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # j dominates i if j is <= in both objectives and < in at least one
            if (
                costs[j] <= costs[i]
                and ginis[j] <= ginis[i]
                and (costs[j] < costs[i] or ginis[j] < ginis[i])
            ):
                is_pareto[i] = False
                break
    return [i for i in range(n) if is_pareto[i]]


pareto_indices = compute_pareto_frontier(avg_costs, avg_ginis)

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Classify points by constraint satisfaction
for i in range(num_r_max):
    if sat_fractions[i] == 1.0:
        color = "green"
    elif sat_fractions[i] == 0.0:
        color = "red"
    else:
        color = "orange"

    marker = "s" if i in pareto_indices else "+"
    size = 40 if i in pareto_indices else 100
    ax.scatter(avg_costs[i], avg_ginis[i], size, color=color, marker=marker, zorder=3)

# Draw Pareto staircase for Pareto-optimal points
pareto_sorted = sorted(pareto_indices, key=lambda i: avg_costs[i])
for k in range(len(pareto_sorted) - 1):
    i, j = pareto_sorted[k], pareto_sorted[k + 1]
    ax.plot(
        [avg_costs[i], avg_costs[j]],
        [avg_ginis[i], avg_ginis[i]],
        color="blue",
        linewidth=1,
    )
    ax.plot(
        [avg_costs[j], avg_costs[j]],
        [avg_ginis[i], avg_ginis[j]],
        color="blue",
        linewidth=1,
    )

# Label each point with r_max value
labels = [rf"$r_{{max}}$={r}" for r in R_MAX_VALUES]
for i in range(num_r_max):
    ax.annotate(
        labels[i],
        (avg_costs[i], avg_ginis[i]),
        textcoords="offset points",
        xytext=(8, -12),
        fontsize=20,
    )

# Legend
green_patch = mpatches.Patch(color="green", label="Feasible (all seeds)")
orange_patch = mpatches.Patch(color="orange", label="Partially feasible")
red_patch = mpatches.Patch(color="red", label="Infeasible (no seeds)")
handles = [green_patch, orange_patch, red_patch]
# Only include patches that are actually used
used_colors = set()
for i in range(num_r_max):
    if sat_fractions[i] == 1.0:
        used_colors.add("green")
    elif sat_fractions[i] == 0.0:
        used_colors.add("red")
    else:
        used_colors.add("orange")
handles = [h for h in handles if h.get_facecolor()[:3] != (1, 1, 1)]
legend_handles = []
if "green" in used_colors:
    legend_handles.append(green_patch)
if "orange" in used_colors:
    legend_handles.append(orange_patch)
if "red" in used_colors:
    legend_handles.append(red_patch)

ax.legend(handles=legend_handles, fontsize=20, loc="best", framealpha=0.4)

ax.set_ylabel("Gini index", fontsize=26)
ax.set_xlabel("Global service cost", fontsize=26)
ax.tick_params(labelsize=26)
ax.grid(True, which="major", linestyle=":", linewidth=1, color="grey", alpha=0.7)
plt.tight_layout()
if args.save:
    plt.savefig(
        os.path.join(SCRIPT_DIR, f"plots/pareto_costs_gini_{cat}_cat.png"),
        format="png",
    )
plt.show()
