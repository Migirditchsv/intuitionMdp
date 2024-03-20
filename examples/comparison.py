import random
import time

from src import mdp
from src.mdp import MDP
from src.visualization import create_convergence_gif, create_mfpt_gif


def solve_mdp_with_and_without_mfpt(filename, size, goal_number, stochasticity, max_iterations, random_seed=None):
    """
    This function solves an MDP with and without using MFPT and creates a GIF of the convergence for both cases.
    :param filename: Prefix for the filenames of the GIFs
    :param max_iterations: Maximum number of iterations for the MDP to solve
    """
    # Experiment randomization
    if random_seed is None:
        mdp.random_seed = random.randint(0, 1000000)

    # # Construct benchmark MDPs
    normal_mdp, mfpt_mdp = generate_benchmark_mdps(size, stochasticity, goal_number, random_seed)

    # # # # Solve the MDP without using MFPT
    # normal_convergence_data, _, _ = normal_mdp.solve(max_iterations, export_convergence_frames=True)
    #
    # # Create a GIF of the normal convergence
    # create_convergence_gif(filename + '_normal', normal_convergence_data, normal_mdp.get_world_model())
    # #create_runtime_gif(normal_convergence_data, filename + '_runtime_normal.gif')

    # Solve the MDP using MFPT
    mftp_convergence_data, _, _ = mfpt_mdp.solve(max_iterations, export_convergence_frames=True, use_mfpt=True)
    print("COMPLETE")


    # Create a GIF of the convergence
    print("Creating GIF, this may take a while...")
    create_convergence_gif(filename + '_mfpt', mftp_convergence_data, mfpt_mdp.get_world_model())
    create_mfpt_gif("mu_convergence_"+filename + '_mfpt', mftp_convergence_data, mfpt_mdp.get_world_model())

    #create_runtime_gif(mftp_convergence_data, filename + '_runtime_mfpt.gif')

def generate_benchmark_mdps(size, stochasticity, goal_number, random_seed=None):

    # Set the random seed if provided
    if random_seed is None:
        random_seed = random.randint(0, 1000000)

    # Initialize normal mdp object
    normal_mdp = MDP(size,
              stochasticity,
              goal_number,
              use_mfpt=False,
              random_seed=random_seed)

    # Initialize mfpt mdp object
    mftp_mdp = MDP(size,
              stochasticity,
              goal_number,
              use_mfpt=True,
              random_seed=random_seed)

    return normal_mdp, mftp_mdp


# Usage:
size = 25
goal_number = 1
stochasticity = 0.1
max_iterations = 100
# Time and date filename for experiment
filename = time.strftime("%Y%m%d-%H%M%S")
filename = 'comparison_' + filename

solve_mdp_with_and_without_mfpt(filename, size, goal_number, stochasticity, max_iterations)

