import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cmdp.config import R_MAX_VALUES

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _fmt(v):
    if v == int(v):
        return str(int(v))
    s = f"{v:g}"
    if s.startswith("0."):
        s = s[1:]
    return s


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

# All points are feasible; shape for Pareto-optimality
for i in range(num_r_max):
    marker = "s" if i in pareto_indices else "o"
    size = 120 if i in pareto_indices else 40
    ax.scatter(avg_costs[i], avg_ginis[i], size, color="green", marker=marker, zorder=3)

# Draw Pareto staircase
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

# Per-point label offsets (offset points: x=right/left, y=up/down)
label_offsets = {
    R_MAX_VALUES.index(0.075): (8, -6),
    R_MAX_VALUES.index(0.10): (8, 6),
    R_MAX_VALUES.index(0.125): (8, 6),
    R_MAX_VALUES.index(0.15): (8, 6),
    R_MAX_VALUES.index(0.20): (-25, -28),
    R_MAX_VALUES.index(0.25): (12, 12),
}

labels = [rf"$r_{{max}}$={_fmt(r)}" for r in R_MAX_VALUES]

# Label all Pareto-optimal points
for i in pareto_indices:
    xytext = label_offsets.get(i, (8, -12))
    ax.annotate(
        labels[i],
        (avg_costs[i], avg_ginis[i]),
        textcoords="offset points",
        xytext=xytext,
        fontsize=20,
    )

# Label the r_max=0.05 dominated point (to the left since it sits at the right edge)
idx_005 = R_MAX_VALUES.index(0.05)
ax.annotate(
    labels[idx_005],
    (avg_costs[idx_005], avg_ginis[idx_005]),
    textcoords="offset points",
    xytext=(-90, -12),
    fontsize=20,
)

# Legend: shape only; list all dominated r_max values using newlines to avoid width expansion
dominated_indices = [i for i in range(num_r_max) if i not in pareto_indices]
dominated_vals = [_fmt(R_MAX_VALUES[i]) for i in dominated_indices]
chunk = 4
lines = [
    ", ".join(dominated_vals[i : i + chunk])
    for i in range(0, len(dominated_vals), chunk)
]
dominated_label = "Dominated\n(" + ",".join(lines) + ")"

pareto_marker = plt.scatter(
    [], [], marker="s", color="green", s=120, label="Pareto-optimal"
)
dominated_marker = plt.scatter(
    [], [], marker="o", color="green", s=40, label=dominated_label
)
ax.legend(
    handles=[pareto_marker, dominated_marker], fontsize=20, loc="best", framealpha=0.4
)

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
