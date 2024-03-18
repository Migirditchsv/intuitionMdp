import pickle
from datetime import datetime
import os
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np

from src.generate_initial_state import generate_null_policy
from src.policy_value_iteration import policy_iteration_step
from src.visualization import plot_value_and_policy


def to_binary(filename, experiment_index, iteration, world_model, value_array, max_delta_value, optimal_value):
    with open(filename, 'ab') as f:
        pickle.dump((experiment_index, iteration, world_model, value_array, max_delta_value, optimal_value), f)


def from_binary(filename):
    experiments = []
    with open(filename, 'rb') as f:
        while True:
            try:
                experiment_data = pickle.load(f)
                experiment = {
                    'experiment_index': experiment_data[0],
                    'iteration': experiment_data[1],
                    'world_model': experiment_data[2],
                    'value_array': experiment_data[3],
                    'max_delta_value': experiment_data[4],
                    'optimal_value': experiment_data[5]
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

def create_heatmap_gifs(filename, frame_duration):
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