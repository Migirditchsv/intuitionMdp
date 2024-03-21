import io
import os

import imageio
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt, colors
from matplotlib.patches import FancyArrowPatch

from src.file_writer import from_binary
from src.generate_initial_state import generate_null_policy
from src.policy_value_iteration import policy_iteration_step
import glob


def create_mfpt_gif(filename, convergence_data, world_model):
    # File structure check
    folder_path = f"./data/frames"
    # Make sure the ./data folder exists, if not create it
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Make sure the ./data/frames folder exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # If frames is not empty, delete all files in the folder
    files = glob.glob('./data/frames/*.png')
    for file in files: # Kind of dangerous be careful lol! the .png helps a little
        os.remove(file)
    # Create the output path
    output_path = f"./data/{filename}.gif"

    # Iterate through the convergence data to create mfpt heatmaps
    for iteration in convergence_data.keys():
        # Progress update:
        print(f"Creating mfpt heatmap for iteration {iteration}: {round(iteration / len(convergence_data) * 100)}% complete")
        policy_grid = convergence_data[iteration]['mfpt_array']
        fig, ax = plot_mu_matrix(policy_grid, iteration, world_model)
        plt.savefig(f"./data/frames/{iteration}_heatmap.png")
        plt.close(fig)

    # Get all the PNG files from the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort files numerically
    # Define a custom sorting key
    def numeric_sort_key(file_name):
        # Extract the number from the filename
        number = int(file_name.split('_')[0])
        return number
    png_files.sort(key=numeric_sort_key)

    # Create a list to hold the images
    images = []

    # Load each file into the images list
    for file_name in png_files:
        file_path = os.path.join(folder_path, file_name)
        images.append(imageio.imread(file_path))

    # Save the images as a gif
    imageio.mimsave(output_path, images, fps= 5) # Adding loop=0 # loop=0 argument makes the gif loop indefinitely

def create_convergence_gif(filename, convergence_data, world_model):
    # File structure check
    folder_path = f"./data/frames"
    # Make sure the ./data folder exists, if not create it
    if not os.path.exists('./data'):
        os.makedirs('./data')
    # Make sure the ./data/frames folder exists, if not create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # If frames is not empty, delete all files in the folder
    files = glob.glob('./data/frames/*.png')
    for file in files: # Kind of dangerous be careful lol! the .png helps a little
        os.remove(file)
    # Create the output path
    output_path = f"./data/{filename}.gif"

    # Iterate through the convergence data to create policy and value heatmaps
    for iteration in convergence_data.keys():
        # Progress update:
        print(f"Creating policy-value heatmap for iteration {iteration}: {round(iteration / len(convergence_data) * 100)}% complete")
        policy_grid = convergence_data[iteration]['policy_array']
        value_array = convergence_data[iteration]['value_array']
        fig, ax = plot_value_and_policy(value_array, policy_grid, iteration, world_model)
        plt.savefig(f"./data/frames/{iteration}_heatmap.png")
        plt.close(fig)

    # Get all the PNG files from the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]

    # Sort files numerically
    # Define a custom sorting key
    def numeric_sort_key(file_name):
        # Extract the number from the filename
        number = int(file_name.split('_')[0])
        return number
    png_files.sort(key=numeric_sort_key)

    # Create a list to hold the images
    images = []

    # Load each file into the images list
    for file_name in png_files:
        file_path = os.path.join(folder_path, file_name)
        images.append(imageio.imread(file_path))

    # Save the images as a gif
    imageio.mimsave(output_path, images, fps= 5) # Adding loop=0 # loop=0 argument makes the gif loop indefinitely

def convergence_gif_from_pickle(filename, frame_duration):
    # Load the data from the binary file
    experiment_index = None
    data = from_binary(filename)
    # For each unique experiment number, create a gif of the value functions converging as heat maps
    for data_point in data:
        # If new experiment has begun, reset the gif counter and filename
        if data_point['experiment_index'] != experiment_index:
            experiment_index = data_point['experiment_index']
            gif_filename = f"{filename}_exp{experiment_index}.gif"
            writer = imageio.get_writer(gif_filename, mode='I', duration=frame_duration, loop=0)

        # Rehydrate the policy grid from the value array
        policy_grid, _ = policy_iteration_step(data_point['world_model'],
                                               data_point['value_array'],
                                               generate_null_policy(data_point['value_array']),
                                               gamma=0.9
                                               )
        fig, ax = plot_value_and_policy(data_point['value_array'],
                                        policy_grid,
                                        data_point['iteration'],
                                        data_point['world_model']
                                        )
        # Save the figure as an image
        plt.savefig("heatmap.png")
        # Close the figure
        plt.close(fig)
        image = imageio.imread("heatmap.png")
        writer.append_data(image)

    writer.close()
    os.remove("heatmap.png")


# Plot the value and policy grids as a heatmap with arrows
def plot_value_and_policy(value_grid, policy_grid, iteration, world_model):
    # Close all figures to prevent memory leaks
    plt.close('all')
    # Pull in the initial state of the world
    state_map = world_model.get_world_map()
    wall_value = world_model.get_wall_value()
    size = value_grid.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use seaborn's 'rocket' color scheme for the heatmap, excluding wall cells
    mask = np.zeros_like(value_grid, dtype=bool)
    mask[value_grid == world_model.get_wall_value()] = True  # Mask wall cells to keep them black

    heatmap = sns.heatmap(value_grid, mask=mask, cmap='viridis', cbar=True, ax=ax,
                cbar_kws={'label': 'Value'}, square=True, linewidths=.5, annot=False)

    # Manually give wall cells a semi-transparent solid rectangle and a black outline
    for i in range(size):
        for j in range(size):
            if state_map[i, j] == world_model.get_wall_value():
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, facecolor='black', alpha=0.5, edgecolor='black', lw=2))
            elif value_grid[i, j] <0 and value_grid[i, j] != wall_value:
                value_grid[i, j] = 0 # Set negative values to 0 for better visualization

    # Overlay goals and policy arrows, adjusting arrow size
    arrow_scale = min(size / 50.0, 0.1)  # Scale down arrow size
    for i in range(size):
        for j in range(size):
            if state_map[i, j] == world_model.get_goal_value():  # Mark goal with G and a green border
                ax.text(j + 0.5, i + 0.5, 'G', ha='center', va='center', color='green', fontsize=12, fontweight='bold')
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=None, edgecolor='green', lw=2))
            elif state_map[i, j] != wall_value:  # Exclude walls
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

    return fig, ax

def plot_transition_matrix(transition_matrix):
    plt.figure(figsize=(8, 8))
    sns.heatmap(transition_matrix, cmap='vlag', annot=False, fmt=".2f")
    plt.title('Transition Matrix Heatmap')
    plt.xlabel('State j')
    plt.ylabel('State i')
    plt.show()


# Plots the mean first passage time matrix as a heatmap. NOTE: If getting weird results, try using a less stochasticity.
# MFPT can fail to converge for moderate stochasticity values if the penalty for hitting a wall is too high.
def plot_mu_matrix(mfpt_grid, iteration=None, world_model=None):
    # Close all figures to prevent memory leaks
    plt.close('all')
    # Pull in the initial state of the world
    state_map = world_model.get_world_map()
    size = mfpt_grid.shape[0]
    fig, ax = plt.subplots(figsize=(8, 8))
    # Use seaborn's 'rocket' color scheme for the heatmap, excluding wall cells
    mask = np.zeros_like(mfpt_grid, dtype=bool)
    mask[state_map == world_model.get_wall_value()] = True  # Mask wall cells to keep them black

    heatmap = sns.heatmap(mfpt_grid, mask=mask, cmap='viridis', cbar=True, ax=ax,
                          cbar_kws={'label': 'Value'}, square=True, linewidths=.5, annot=False)

    # Manually give wall cells a semi-transparent solid rectangle and a black outline
    for i in range(size):
        for j in range(size):
            if state_map[i, j] == world_model.get_wall_value():
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=True, facecolor='black', alpha=0.5, edgecolor='black', lw=2))

    # Overlay goals and policy arrows, adjusting arrow size
    arrow_scale = min(size / 50.0, 0.1)  # Scale down arrow size
    for i in range(size):
        for j in range(size):
            if state_map[i, j] == world_model.get_goal_value():  # Mark goal with G and a green border
                ax.text(j + 0.5, i + 0.5, 'G', ha='center', va='center', color='green', fontsize=12,
                        fontweight='bold')
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=None, edgecolor='green', lw=2))

    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.set_xticklabels(range(1, size + 1))
    ax.set_yticklabels(range(1, size + 1))
    ax.set_title('MFPT\nIteration: ' + str(iteration))

    return fig, ax