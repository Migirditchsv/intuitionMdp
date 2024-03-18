import pickle
from datetime import datetime


def to_binary(filename, sample_number, world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity,
              optimal_value):
    with open(filename, 'wb') as f:
        pickle.dump((sample_number, world_seed, size, value_array, max_delta_value, solver_iteration, optimal_value), f)


def from_binary(filename):
    with (open(filename, 'rb') as f):
        sample_number, world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity, optimal_value = \
            pickle.load(f)
    return sample_number, world_seed, size, value_array, max_delta_value, solver_iteration, stochasticity, optimal_value


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
