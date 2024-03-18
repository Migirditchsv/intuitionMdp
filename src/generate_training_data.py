import sys

import file_writer as fw
import mdp
import random

def generate_mdps_and_solutions(max_experiment_number, size, stochasticity, goal_number=1, use_mfpt=False):
    random_seed = 42  # Random seed for generating DIFFERENT random seeds so each MDP has a
    # derterministically unique world
    # Generate a unique file name for this experiment
    data_filename = fw.generate_experiment_name(size, stochasticity, use_mfpt)
    for experiment_index in range(max_experiment_number):
        # Create a unique world seed for each MDP
        world_seed = random.randint(0, sys.maxsize)
        mdp_instance = mdp.MDP(size, stochasticity, goal_number, use_mfpt, world_seed)
        # Initialize variables to store the solution
        value_array = None
        max_delta_value = None
        iteration = None
        optimal_value = None

        # Solve the MDP
        convergence_data, optimal_value = mdp_instance.solve()

        for iteration in convergence_data.keys():
            # Parse the solution out of the solution dictionary
            value_array = convergence_data[iteration]['value_array']
            max_delta_value = convergence_data[iteration]['max_delta_value']


        # Save the current state and solution to a binary file if the data looks healthy
        if not any( element is None for element in [size, value_array, max_delta_value, iteration,
                                                    stochasticity, optimal_value]):
            fw.to_binary(data_filename, experiment_index, world_seed, size, value_array, max_delta_value, iteration,
                         stochasticity, optimal_value)

    print(f"Generated {max_experiment_number} MDPs and solutions and saved to {data_filename}")

    return data_filename

# Call the function to generate and solve MDPs
max_experiment_number = 3
size = 5
stochasticity = 0.1
filename = generate_mdps_and_solutions(max_experiment_number, size, stochasticity)
# Call the function to create the gifs
fw.create_heatmap_gifs(filename)
