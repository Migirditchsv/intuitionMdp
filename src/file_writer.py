import pickle
from datetime import datetime
import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np


def to_binary(filename, experiment_index, world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity,
              optimal_value):
    with open(filename, 'ab') as f:
        pickle.dump((experiment_index, world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity,
                     optimal_value), f)


def from_binary(filename):
    experiments = []
    with open(filename, 'rb') as f:
        while True:
            try:
                experiment_data = pickle.load(f)
                experiment = {
                    'experiment_index': experiment_data[0],
                    'world_seed': experiment_data[1],
                    'size': experiment_data[2],
                    'value_array': experiment_data[3],
                    'max_delta_value': experiment_data[4],
                    'solver_iteration': experiment_data[5],
                    'stochasticity': experiment_data[6],
                    'optimal_value': experiment_data[7]
                }
                experiments.append(experiment)
            except EOFError:
                break
    return experiments

def generate_experiment_name(size, stochasticity, use_mfpt):
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    date_time = now.strftime("%Y%m%d_%H%M%S")

    # Format the use_mfpt as a string
    use_mfpt_str = 'mfpt' if use_mfpt else 'no_mfpt'

    # Create the file name
    filename = f"experiment_{date_time}_size{size}_stoch{stochasticity}_{use_mfpt_str}.bin"

    return filename


def create_heatmap_gifs(filename):
    # Load the data from the binary file
    data = from_binary(filename)

    # Group the data by unique sample numbers
    grouped_data = {}
    for sample_number, world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity, optimal_value in data:
        if sample_number not in grouped_data:
            grouped_data[sample_number] = []
        grouped_data[sample_number].append(
            (world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity, optimal_value))

    # For each unique sample number, create N gifs of the value functions converging as heat maps
    for sample_number, experiments in grouped_data.items():
        for i in range(len(experiments)):
            world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity, optimal_value = \
            experiments[i]

            # Create a gif of the value functions converging as a heat map
            gif_filename = f"{filename}_{sample_number}_{i}.gif"
            with imageio.get_writer(gif_filename, mode='I') as writer:
                for value in value_array:
                    fig, ax = plt.subplots()
                    heatmap = ax.imshow(value, cmap='hot', interpolation='nearest')
                    plt.colorbar(heatmap)
                    plt.savefig("heatmap.png")
                    plt.close(fig)
                    image = imageio.imread("heatmap.png")
                    writer.append_data(image)
            os.remove("heatmap.png")
