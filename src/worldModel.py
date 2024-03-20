import numpy as np

from src.generate_initial_state import generate_world_map, generate_simple_initial_value


class WorldModel:
    def __init__(self, size, stochasticity, goal_number, wall_clustering, density, random_seed):
        self.wall_clustering = wall_clustering
        self.density = density
        self.wall_reward = -1
        self.goal_reward = 1
        self.stationary_reward = 0.0
        self.movement_reward = -0.01

        self.wall_value = -0.50
        self.goal_value = 1.0
        self.empty_value = 0.0
        self.random_seed = random_seed

        self.size = size
        self.stochasticity = stochasticity
        self.goal_number = goal_number
        self.world_map = generate_world_map(size, goal_number, density, wall_clustering, self.wall_value, self.goal_value,
                                            self.empty_value, self.random_seed)
        self.action_space = {
            'up': (-1, 0), 'right': (0, 1), 'down': (1, 0), 'left': (0, -1),
            'up-left': (-1, -1), 'up-right': (-1, 1), 'down-left': (1, -1), 'down-right': (1, 1), 'stay': (0, 0)
        }

    # Getter methods
    def get_wall_reward(self):
        return self.wall_reward

    def get_goal_reward(self):
        return self.goal_reward

    def get_stationary_reward(self):
        return self.stationary_reward

    def get_movement_reward(self):
        return self.movement_reward

    def get_wall_value(self):
        return self.wall_value

    def get_goal_value(self):
        return self.goal_value

    def get_empty_value(self):
        return self.empty_value

    def get_random_seed(self):
        return self.random_seed

    def get_size(self):
        return self.size

    def get_stochasticity(self):
        return self.stochasticity

    def get_goal_number(self):
        return self.goal_number

    def get_world_map(self):
        return self.world_map.copy()

    def get_action_space(self):
        return self.action_space.copy()

    # Setter methods
    def set_wall_reward(self, wall_reward):
        self.wall_reward = wall_reward

    def set_goal_reward(self, goal_reward):
        self.goal_reward = goal_reward

    def set_stationary_reward(self, stationary_reward):
        self.stationary_reward = stationary_reward

    def set_movement_reward(self, movement_reward):
        self.movement_reward = movement_reward

    def set_wall_value(self, wall_value):
        self.wall_value = wall_value

    def set_goal_value(self, goal_value):
        self.goal_value = goal_value

    def set_empty_value(self, empty_value):
        self.empty_value = empty_value

    def set_random_seed(self, random_seed):
        self.random_seed = random_seed

    def set_size(self, size):
        print("WARNING: Changing the size of the world will automatically regenerate the world map")
        self.size = size
        self.world_map = generate_world_map(size, self.goal_number, self.density, self.wall_clustering, self.wall_value,
                                            self.goal_value, self.empty_value, self.random_seed)
    def set_stochasticity(self, stochasticity):
        self.stochasticity = stochasticity

    def set_goal_number(self, goal_number):
        print("WARNING: Changing the number of goals will automatically regenerate the world map")
        self.goal_number = goal_number
        self.world_map = generate_world_map(self.size, goal_number, self.density, self.wall_clustering, self.wall_value,
                                            self.goal_value, self.empty_value, self.random_seed)

    def set_world_map(self, world_map):
        self.world_map = world_map
        # Update the size and goal number
        self.size = len(world_map)
        self.goal_number = np.count_nonzero(self.world_map == self.goal_value) # Count goals
    def set_action_space(self, action_space):
        self.action_space = action_space
        print("WARNING: Changing the action space is not recommended")