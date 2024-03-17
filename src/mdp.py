from src.policy_value_iteration import policy_value_iteration
from src.generate_initial_state import generate_simple_initial_value, generate_initial_values_simplex
from src.mfpt import compute_mfpt
from src.visualization import plot_transition_matrix, plot_value_and_policy
import pickle


class MDP:
    def __init__(self, size, stochasticity, goal_number, use_mfpt, random_seed):
        # Operational Parameters
        self.max_iterations = 100
        self.convergence_threshold = 0.01
        self.random_seed = random_seed
        # World Model Parameters
        self.size = size
        self.stochasticity = stochasticity
        self.goal_number = goal_number
        self.use_mfpt = use_mfpt
        self.initial_state = generate_simple_initial_value(size)
        self.iterations = 0
        # Simplex noise parameters for wall placement
        self.scale = size / 10  # Controls the level of detail (smaller values generate larger "blobs" of walls) 1/10th of the world size
        # is a good starting point
        self.octaves = 5  # Adds detail at different scales, values of 3-5 offer a good balance of uniformity and complexity
        self.persistence = 0.5  # Affects the amplitude of each octave. Lower values result in smoother, less pronounced noise,
        # while higher values make each octave's contribution more significant. For moderate density, values around 0.4 to
        # 0.6 are often suitable.
        self.lacunarity = 2.5  # Controls the frequency growth for each octave. A value around 2.0 to 3.0 is typical and can
        # produce a natural-looking pattern.
        self.threshold = 0.5  # The threshold value for wall placement. Higher values result in fewer walls.

    def solve(self):
        value_grid, policy_grid = policy_value_iteration(self.initial_state, self.stochasticity)

        if self.use_mfpt:
            mfpt = compute_mfpt(policy_grid, self.stochasticity)
            plot_transition_matrix(mfpt)

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