import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

# Load TEST data
# Note: We manually construct the list of valid data to handle the empty rows
gini_obj = np.load('results/gini_2_cat_TEST.npy', allow_pickle=True)
reb_obj = np.load('results/cost_reb_2_cat_TEST.npy', allow_pickle=True)
fail_obj = np.load('results/cost_fail_2_cat_TEST.npy', allow_pickle=True)
bikes_obj = np.load('results/cost_bikes_2_cat_TEST.npy', allow_pickle=True)

# Extract only existing betas
valid_betas = []
gini_data = []
reb_data = []
fail_data = []
bikes_data = []

labels = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']

for i in range(11):
    if len(gini_obj[i]) > 0:
        valid_betas.append(labels[i])
        gini_data.append(gini_obj[i])
        reb_data.append(reb_obj[i])
        fail_data.append(fail_obj[i])
        bikes_data.append(bikes_obj[i])

def plot_metric(data, ylabel, filename, color_idx):
    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
    
    # Boxplot
    box_color = sns.color_palette("viridis", 11)[color_idx]
    box = ax.boxplot(data, patch_artist=True, notch=False, vert=True, widths=0.4)
    
    for patch in box['boxes']:
        patch.set_facecolor(box_color)
        patch.set_edgecolor('black')
        patch.set_alpha(0.8)
    for median in box['medians']:
        median.set(color='gold', linewidth=1.5)
        
    ax.set_xlabel(r'$\beta$', fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.set_xticklabels(valid_betas, fontsize=16)
    ax.tick_params(labelsize=16)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f"{filename}.png")
    print(f"Saved {filename}.png")
    # plt.show() # Uncomment to see popup

print("Generating Boxplots...")
plot_metric(gini_data, "Gini index", "boxplot_gini_TEST", 7)
plot_metric(reb_data, "Weighted reb. op's", "boxplot_reb_TEST", 2)
plot_metric(fail_data, "Failure rate (Count)", "boxplot_fail_TEST", 2)
plot_metric(bikes_data, "Number of vehicles", "boxplot_bikes_TEST", 2)