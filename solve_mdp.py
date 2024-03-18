from src.generate_initial_state import generate_null_policy, generate_initial_values_simplex, \
    generate_world_map, generate_simple_initial_value
from src.policy_value_iteration import policy_value_iteration
from src.visualization import plot_value_and_policy, plot_transition_matrix, plot_mu_matrix
from src.mfpt import construct_transition_matrix, compute_mfpt

# Operational Parameters
max_iterations = 25
convergence_threshold = 0.01

# World Model Parameters
size = 25  # Size of the grid
goal_number = 1
stochasticity = 0.3  # Instead of a transition matrix, we use a stochasticity parameter
movement_cost_scale = 0.01  # Scaling factor for movement cost
action_space = {
    'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1),
    'up-left': (-1, -1), 'up-right': (-1, 1), 'down-left': (1, -1), 'down-right': (1, 1), 'stay': (0, 0)
}

# Simple wall placement parameters
density = 0.2  # Probability a cell will be a goal using simple random placement (not used with simplex noise)
wall_clustering = 0.45  # Probability a new wall will be placed adjacent to an existing wall using simple random
# placement (not used with simplex noise)

# Simplex noise parameters for wall placement
scale = size / 10  # Controls the level of detail (smaller values generate larger "blobs" of walls) 1/10th of the world size
# is a good starting point
octaves = 5  # Adds detail at different scales, values of 3-5 offer a good balance of uniformity and complexity
persistence = 0.5  # Affects the amplitude of each octave. Lower values result in smoother, less pronounced noise,
# while higher values make each octave's contribution more significant. For moderate density, values around 0.4 to
# 0.6 are often suitable.
lacunarity = 2.5  # Controls the frequency growth for each octave. A value around 2.0 to 3.0 is typical and can
# produce a natural-looking pattern.
threshold = 0.5 # The threshold value for wall placement. Higher values result in fewer walls.




# Generate the initial value array using Perlin noise
initial_value_array = generate_initial_values_simplex(size, goal_number, scale, octaves, persistence, lacunarity, threshold)

# Example usage with the previously generated initial value array
initial_policy = generate_null_policy(initial_value_array)

# Plot the initial value array and policy
plot_value_and_policy(initial_value_array, initial_policy, 0)

# Run the policy and value iteration algorithm until the delta value is below the threshold
value_array = initial_value_array
policy_array = initial_policy
max_delta_value = float('inf')
iteration_count = 0
while max_delta_value > convergence_threshold and iteration_count < max_iterations:
    # Compute the mean first passage time for the current policy
    mu, t_matrix = compute_mfpt(policy_array, action_space, stochasticity)

    # Run the policy and value iteration algorithm
    value_array, policy_array, max_delta_value = policy_value_iteration(value_array, action_space, stochasticity, movement_cost_scale)
    # Re-plot with Seaborn's styling and the 'rocket' color scheme
    # plot_value_and_policy_seaborn(value_array, policy_array)
    iteration_count += 1

print("Converged to a solution in", iteration_count, "steps")

plot_value_and_policy(value_array, policy_array, iteration_count)
plot_mu_matrix(mu)