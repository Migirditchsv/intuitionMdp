import time
from src.generate_initial_state import  generate_benchmark_mdps
from src.policy_value_iteration import policy_iteration_step,
from src.file_writer import to_binary, create_heatmap_gifs

def solve_mdp_with_and_without_mfpt(filename, max_iterations, frame_duration):
    # Construct benchmark MDPs
    normal_mdp, mfpt_mdp = generate_benchmark_mdps(mdp.size, mdp.stochasticity, mdp.goal_number, mdp.random_seed)

    # Solve the MDP without using MFPT
    normal_mdp.solve(max_iterations)


    # Create a GIF of the convergence
    create_heatmap_gifs(filename, frame_duration)

    # Solve the MDP using MFPT


    # Create a GIF of the convergence
    create_heatmap_gifs(filename, frame_duration)

    return (iteration, time_no_mfpt), (iteration_mfpt, time_mfpt)

# Usage:
mdp = generate_random_mdp()
solve_mdp_with_and_without_mfpt('output_filename', mdp, 1000, 0.1)