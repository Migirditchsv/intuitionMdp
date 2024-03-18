import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors
from matplotlib.patches import FancyArrowPatch

# Plot the value and policy grids as a heatmap with arrows
def plot_value_and_policy(value_grid, policy_grid, iteration, world_model):
    # Pull in the initial state of the world
    state_map = world_model.get_world_map()
    size = value_grid.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use seaborn's 'rocket' color scheme for the heatmap, excluding wall cells
    mask = np.zeros_like(value_grid, dtype=bool)
    mask[value_grid == -1] = True  # Mask wall cells to keep them black

    heatmap = sns.heatmap(value_grid, mask=mask, cmap='rocket', cbar=True, ax=ax,
                cbar_kws={'label': 'Value'}, square=True, linewidths=.5, annot=False)

    # Manually give wall cells a semi-transparent solid rectangle and a black outline
    for i in range(size):
        for j in range(size):
            if state_map[i, j] == world_model.get_wall_value():
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor='black', alpha=0.5, edgecolor='black', lw=2))
            elif value_grid[i, j] <0 and value_grid[i, j] != -1:
                value_grid[i, j] = 0 # Set negative values to 0 for better visualization

    # Overlay goals and policy arrows, adjusting arrow size
    arrow_scale = min(size / 50.0, 0.1)  # Scale down arrow size
    for i in range(size):
        for j in range(size):
            if state_map[i, j] == world_model.get_goal_value():  # Mark goal with G and a green border
                ax.text(j + 0.5, i + 0.5, 'G', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=None, edgecolor='green', lw=2))
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
    plt.figure(figsize=(8, 8))
    sns.heatmap(transition_matrix, cmap='vlag', annot=False, fmt=".2f")
    plt.title('Transition Matrix Heatmap')
    plt.xlabel('State j')
    plt.ylabel('State i')
    plt.show()


# Plots the mean first passage time matrix as a heatmap. NOTE: If getting weird results, try using a less stochasticity.
# MFPT can fail to converge for moderate stochasticity values if the penalty for hitting a wall is too high.
def plot_mu_matrix(mu_matrix):
    plt.figure(figsize=(8, 8))

    # Create a colormap for the heatmap
    cmap = plt.get_cmap('rocket')
    norm = colors.Normalize(vmin=mu_matrix.min(), vmax=mu_matrix.max())

    # Create the heatmap using imshow
    plt.imshow(mu_matrix, cmap=cmap, norm=norm)

    # Add a colorbar
    plt.colorbar(label='Value')

    # Add annotations
    # for i in range(mu_matrix.shape[0]):
    #     for j in range(mu_matrix.shape[1]):
    #         text = plt.text(j, i, np.round(mu_matrix[i, j], 2),
    #                        ha="center", va="center", color="w")


    plt.title('Mu Matrix Heatmap')
    plt.xlabel('State j')
    plt.ylabel('State i')
    plt.show()