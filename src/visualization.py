import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch


def plot_value_and_policy(value_grid, policy_grid, iteration):
    size = value_grid.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use seaborn's 'rocket' color scheme for the heatmap, excluding wall cells
    mask = np.zeros_like(value_grid, dtype=bool)
    mask[value_grid == -1] = True  # Mask wall cells to keep them black

    sns.heatmap(value_grid, mask=mask, cmap='rocket', cbar=True, ax=ax,
                cbar_kws={'label': 'Value'}, square=True, linewidths=.5, annot=False)

    # Manually color wall cells black
    for i in range(size):
        for j in range(size):
            if value_grid[i, j] == -1:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color='grey'))
            elif value_grid[i, j] <0 and value_grid[i, j] != -1:
                value_grid[i, j] = 0 # Set negative values to 0 for better visualization

    # Overlay goals and policy arrows, adjusting arrow size
    arrow_scale = min(size / 50.0, 0.1)  # Scale down arrow size
    for i in range(size):
        for j in range(size):
            if value_grid[i, j] == 1:  # Mark goal with G
                ax.text(j + 0.5, i + 0.5, 'G', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
            elif value_grid[i, j] != -1:  # Exclude walls
                dx, dy = policy_grid[i, j]
                if dx != 0 or dy != 0:
                    # Adjust arrow size to not exceed cell size
                    ax.add_patch(
                        FancyArrowPatch((j + 0.5, i + 0.5), (j + 0.5 + arrow_scale * dy, i + 0.5 + arrow_scale * dx),
                                        arrowstyle='->', color='cyan', mutation_scale=20, linewidth=2))

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(1, size + 1))
    ax.set_yticklabels(range(1, size + 1))
    ax.set_title('Iteration: ' + str(iteration))

    plt.show()

def plot_transition_matrix(transition_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(transition_matrix, cmap='vlag', annot=False, fmt=".2f")
    plt.title('Transition Matrix Heatmap')
    plt.xlabel('State j')
    plt.ylabel('State i')
    plt.show()

def plot_mu_matrix(mu_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(mu_matrix, cmap='rocket', annot=True, fmt=".2f")
    plt.title('Mu Matrix Heatmap')
    plt.xlabel('State j')
    plt.ylabel('State i')
    plt.show()