from generate_initial_state import generate_initial_values, generate_null_policy_fixed, generate_initial_values_simplex
from policy_value_iteration import policy_value_iteration
from visualization import plot_value_and_policy

# Operational Parameters
max_iterations = 100
convergence_threshold = 0.001

# World Model Parameters
size = 25
goal_number = 5

# Simple wall placement parameters
density = 0.2  # Probability a cell will be a goal using simple random placement (not used with simplex noise)
wall_clustering = 0.45  # Probability a new wall will be placed adjacent to an existing wall using simple random
# placement (not used with simplex noise)

# Simplex noise parameters for wall placement
scale = 0.5  # Controls the level of detail (smaller values generate larger "blobs" of walls) 1/10th of the world size
# is a good starting point
octaves = 5  # Adds detail at different scales, values of 3-5 offer a good balance of uniformity and complexity
persistence = 1.0  # Affects the amplitude of each octave. Lower values result in smoother, less pronounced noise,
# while higher values make each octave's contribution more significant. For moderate density, values around 0.4 to
# 0.6 are often suitable.
lacunarity = 3.0  # Controls the frequency growth for each octave. A value around 2.0 to 3.0 is typical and can
# produce a natural-looking pattern.
threshold = 0.1  # The threshold value for wall placement. Higher values result in fewer walls.
# MDP Parameters
stochasticity = 0.5  # Instead of a transition matrix, we use a stochasticity parameter
movement_cost_scale = 5.0  # Scaling factor for movement cost

# Generate the initial value array using Perlin noise
initial_value_array = generate_initial_values_simplex(size, goal_number, scale, octaves, persistence, lacunarity, threshold)

# Example usage with the previously generated initial value array
initial_policy = generate_null_policy_fixed(initial_value_array)

# Plot the initial value array and policy
# plot_value_and_policy(initial_value_array, initial_policy, 0)

# Run the policy and value iteration algorithm until the delta value is below the threshold
value_array = initial_value_array
max_delta_value = float('inf')
iteration_count = 0
while max_delta_value > convergence_threshold and iteration_count < max_iterations:
    # Run the policy and value iteration algorithm
    value_array, policy_array, max_delta_value = policy_value_iteration(value_array, stochasticity, movement_cost_scale)
    # Re-plot with Seaborn's styling and the 'rocket' color scheme
    # plot_value_and_policy_seaborn(value_array, policy_array)
    iteration_count += 1

print("Converged to a solution in", iteration_count, "steps")

plot_value_and_policy(value_array, policy_array, iteration_count)
