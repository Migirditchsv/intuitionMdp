from src.policy_value_iteration import policy_value_iteration, value_iteration_step, policy_iteration_step, \
    policy_iteration_mfpt_step
from src.generate_initial_state import generate_simple_initial_value, generate_initial_values_simplex, \
    generate_null_policy_fixed, generate_world_map
from src.mfpt import compute_mfpt
from src.visualization import plot_transition_matrix, plot_value_and_policy, plot_mu_matrix
import pickle
import numpy as np

from src.worldModel import WorldModel


class MDP:
    def __init__(self, size, stochasticity, goal_number, use_mfpt, random_seed):
        ############################
        # Operational Parameters
        ############################
        self.policy_unstable = True
        self.iteration_count = 0
        self.max_iterations = 1000
        self.convergence_threshold = 1.0 / 10 ** (size * size)
        self.random_seed = random_seed
        ############################
        # MFPT Parameters
        ############################
        self.gamma = 0.9  # Discount factor
        self.use_mfpt = use_mfpt
        self.iterations_per_mfpt_update = 3  # Debnaith et al. (2019) suggest 3 iterations per MFPT computation
        ############################
        # World Model Parameters
        ############################
        self.size = size
        self.stochasticity = stochasticity
        self.goal_number = goal_number
        # Parameters for simple wall placement
        self.density = 0.2  # Probability a cell will be a wall using simple random placement (not used with simplex
        # noise)
        self.wall_clustering = 0.35  # Probability a new wall will be placed adjacent to an existing wall using
        # simple random
        # Simplex noise parameters for wall placement
        self.scale = size / 10  # Controls the level of detail (smaller values generate larger "blobs" of walls)
        # 1/10th of the world size
        # is a good starting point
        self.octaves = 5  # Adds detail at different scales, values of 3-5 offer a good balance of uniformity and
        # complexity
        self.persistence = 0.5  # Affects the amplitude of each octave. Lower values result in smoother,
        # less pronounced noise, while higher values make each octave's contribution more significant. For moderate
        # density, values around 0.4 to 0.6 are often suitable.
        self.lacunarity = 2.5  # Controls the frequency growth for each octave. A value around 2.0 to 3.0 is typical
        # and can
        # produce a natural-looking pattern.
        self.threshold = 0.5  # The threshold value for wall placement. Higher values result in fewer walls.

        ############################
        # Init World Model
        ############################
        self.world_model = WorldModel(self.size, self.stochasticity, self.goal_number, self.wall_clustering, self.density)

        ############################
        # Tracking & solution variables
        ############################
        self.iterations = 0
        self.value_array = self.world_model.get_world_map()
        self.policy_array = generate_null_policy_fixed(self.value_array)
        self.update_states = [index for index, value in np.ndenumerate(self.world_model.get_world_map())]

    def solve(self):
        # Plot initial world map
        plot_value_and_policy(self.value_array, self.policy_array, self.iteration_count, self.world_model)
        # Generate the initial policy array as a null policy, stationary.
        policy_array = generate_null_policy_fixed(self.world_model.get_world_map())
        max_delta_value = float('inf')
        self.iteration_count = 0
        while max_delta_value > self.convergence_threshold and self.policy_unstable and self.iteration_count < self.max_iterations:
            # 1) Update the value array under the current policy
            self.value_array, max_delta_value = value_iteration_step(self.world_model,
                                                                     self.value_array,
                                                                     self.policy_array,
                                                                     self.gamma,
                                                                     self.update_states)
            # 2) Update the policy array based on the updated value array
            self.policy_array, self.policy_unstable = policy_iteration_step(self.world_model,
                                                                            self.value_array,
                                                                            self.policy_array,
                                                                            self.gamma,
                                                                            self.update_states)
            # 3) Compute the mean first passage time for the current policy, every few iterations. Update the policy to
            # minimize the mean first passage time.
            if self.use_mfpt and self.iteration_count % self.iterations_per_mfpt_update == 0:
                self.mfpt_array, t_matrix = compute_mfpt(policy_array, self.world_model)
                self.policy_array = policy_iteration_mfpt_step(self.world_model, self.value_array, self.policy_array, self.mfpt_array)
            # Update the iterations
            self.iteration_count += 1
            # Plots for debugging
            # plot_value_and_policy(self.value_array, self.policy_array, self.iteration_count, self.world_model)
            # if self.use_mfpt:
            #     self.mfpt_array, t_matrix = compute_mfpt(policy_array, self.world_model)
            #     plot_mu_matrix(self.mfpt_array)
            #     plot_transition_matrix(t_matrix)
        print("Converged to a solution in", self.iteration_count, "steps")

        plot_value_and_policy(self.value_array, self.policy_array, self.iteration_count, self.world_model)
        if self.use_mfpt:
            self.mfpt_array, t_matrix = compute_mfpt(policy_array, self.world_model)
            plot_mu_matrix(self.mfpt_array)
            plot_transition_matrix(t_matrix)

    # Write data to a pickle file
    def write_to_file(self, filename):
        # Open the file in write binary mode
        with open('data/' + filename, 'wb') as file:
            # Write the data to the file
            pickle.dump((self.iteration_number, self.initial_value, self.current_value, self.current_policy,
                         self.current_mfpt), file)

    # Getter methods
    def get_size(self):
        return self.size

    def get_stochasticity(self):
        return self.stochasticity

    def get_goal_number(self):
        return self.goal_number

    def get_use_mfpt(self):
        return self.use_mfpt

    def get_scale(self):
        return self.scale

    def get_octaves(self):
        return self.octaves

    def get_persistence(self):
        return self.persistence

    def get_lacunarity(self):
        return self.lacunarity

    def get_threshold(self):
        return self.threshold

    def get_density(self):
        return self.density

    def get_wall_clustering(self):
        return self.wall_clustering

    def get_normal_iterations_per_mfpt(self):
        return self.iterations_per_mfpt_update

    def get_action_space(self):
        return self.action_space

    # Setter methods
    def set_size(self, size):
        self.size = size

    def set_stochasticity(self, stochasticity):
        self.stochasticity = stochasticity

    def set_goal_number(self, goal_number):
        self.goal_number = goal_number

    def set_use_mfpt(self, use_mfpt):
        self.use_mfpt = use_mfpt

    def set_scale(self, scale):
        self.scale = scale

    def set_octaves(self, octaves):
        self.octaves = octaves

    def set_persistence(self, persistence):
        self.persistence = persistence

    def set_lacunarity(self, lacunarity):
        self.lacunarity = lacunarity

    def set_threshold(self, threshold):
        self.threshold = threshold

    def set_density(self, density):
        self.density = density

    def set_wall_clustering(self, wall_clustering):
        self.wall_clustering = wall_clustering

    def set_normal_iterations_per_mfpt(self, normal_iterations_per_mfpt):
        self.iterations_per_mfpt_update = normal_iterations_per_mfpt

    def set_action_space(self, action_space):
        self.action_space = action_space
