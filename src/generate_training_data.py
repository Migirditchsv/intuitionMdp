import sys

import io
import mdp
import random

def generate_mdps_and_solutions(sample_number, size, stochasticity, goal_number=1, use_mfpt=False):
    random_seed = 42  # Random seed for generating DIFFERENT random seeds so each MDP has a
    # derterministically unique world
    # Generate a unique file name for this experiment
    filename = io.generate_experiment_name(size, stochasticity, use_mfpt)
    for i in range(sample_number):
        # Create a unique world seed for each MDP
        world_seed = random.randint(0, sys.maxsize)
        mdp_instance = mdp.MDP(size, stochasticity, goal_number, use_mfpt, world_seed)
        # Initialize variables to store the solution
        value_array = None
        max_delta_value = None
        iteration = None
        optimal_value = None

        # Solve the MDP
        convergence_data, optimal_value = mdp.solve()

        for iteration in convergence_data.keys():
            # Parse the solution out of the solution dictionary
            value_array = convergence_data[iteration]['value_array']
            max_delta_value = convergence_data[iteration]['max_delta_value']


        # Save the current state and solution to a binary file if the data looks healthy
        if not any( element is None for element in [size, value_array, max_delta_value, iteration,
                                                    stochasticity, optimal_value]):
            io.to_binary(filename, sample_number, world_seed, size, value_array, max_delta_value, iteration,
                         stochasticity, optimal_value)

# Call the function to generate and solve MDPs
generate_mdps_and_solutions(3, 5, 0.1)