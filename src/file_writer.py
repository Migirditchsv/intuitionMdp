import pickle
from datetime import datetime


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

