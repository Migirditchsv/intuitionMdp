from src.mdp import MDP

# Define parameters for the MDP
size = 15
stochasticity = 0.01
goal_number = 1
use_mfpt = False
random_seed = 42

# Create an instance of the MDP class
mdp_instance = MDP(size, stochasticity, goal_number, use_mfpt, random_seed)

# Solve the MDP
mdp_instance.solve()