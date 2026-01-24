import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

gini_obj = np.load('results/gini_2_cat_TEST.npy', allow_pickle=True)
cost_obj = np.load('results/cost_2_cat_TEST.npy', allow_pickle=True)

# Extract Averages
avg_ginis = []
avg_costs = []
labels = []
betas = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

for i in range(11):
    if len(gini_obj[i]) > 0:
        avg_ginis.append(np.mean(gini_obj[i]))
        avg_costs.append(np.mean(cost_obj[i]))
        labels.append(r'$\beta$=' + betas[i])

sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Plot Points
ax.scatter(avg_costs, avg_ginis, 60, color='blue', marker='s', label='Pareto efficient')

# Connect lines between existing points
for i in range(len(avg_costs) - 1):
    # Draw "Manhattan" style steps like the original paper
    ax.plot([avg_costs[i], avg_costs[i+1]], [avg_ginis[i], avg_ginis[i]], color='blue', linewidth=1)
    ax.plot([avg_costs[i+1], avg_costs[i+1]], [avg_ginis[i], avg_ginis[i+1]], color='blue', linewidth=1)

# Add Labels
for i, txt in enumerate(labels):
    plt.annotate(txt, (avg_costs[i], avg_ginis[i]), xytext=(0, 10), 
                 textcoords='offset points', ha='center', fontsize=12)

ax.set_ylabel('Gini index', fontsize=20)
ax.set_xlabel('Global service cost', fontsize=20)
ax.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=14)

plt.tight_layout()
plt.savefig(f"pareto_TEST.png")
print("Saved pareto_TEST.png")