import random
import numpy as np
from noise import snoise2


def generate_initial_values(size, goal_number, density, wall_clustering):
    # Initialize the grid
    grid = np.zeros((size, size))

    # Place goal states randomly
    goals_placed = 0
    while goals_placed < goal_number:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if grid[x, y] == 0:  # Ensure not placing a goal on top of another goal
            grid[x, y] = 1
            goals_placed += 1

    # Calculate the number of walls to be placed
    total_cells = size * size
    wall_cells = int(total_cells * density)

    # Place walls
    walls_placed = 0
    while walls_placed < wall_cells:
        if walls_placed == 0:  # Place the first wall randomly
            x, y = random.randint(0, size - 1), random.randint(0, size - 1)
            if grid[x, y] == 0:  # Ensure not placing a wall on a goal
                grid[x, y] = -1
                walls_placed += 1
        else:
            # Try to cluster walls based on wall_clustering probability
            if random.random() <= wall_clustering:
                # Find an existing wall and place a new wall adjacent to it
                existing_wall = random.choice(np.argwhere(grid == -1))
                directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
                random.shuffle(directions)  # Shuffle directions to randomize wall placement
                for dx, dy in directions:
                    new_x, new_y = existing_wall[0] + dx, existing_wall[1] + dy
                    if 0 <= new_x < size and 0 <= new_y < size and grid[new_x, new_y] == 0:
                        grid[new_x, new_y] = -1
                        walls_placed += 1
                        break  # Exit after placing one wall
            else:
                # Place a wall randomly (not adjacent to an existing wall)
                x, y = random.randint(0, size - 1), random.randint(0, size - 1)
                if grid[x, y] == 0:  # Ensure not placing a wall on a goal
                    grid[x, y] = -1
                    walls_placed += 1

    return grid


import numpy as np


# Assuming the 'noise' module is available, let's mock a basic noise function for demonstration
def pnoise2(x, y, octaves=1, persistence=0.5, lacunarity=2.0, repeatx=1024, repeaty=1024, base=0.0):
    # This is a placeholder function to simulate Perlin noise values.
    # The actual Perlin noise function would generate continuous, smooth, random-like values.
    return np.random.rand()


def generate_initial_values_perlin_noise(size, goal_number, scale, octaves, persistence, lacunarity):
    grid = np.zeros((size, size))

    # Place goal states randomly
    for _ in range(goal_number):
        while True:
            x, y = np.random.randint(size, size=2)
            if grid[x, y] == 0:
                grid[x, y] = 1
                break

    # Generate Perlin noise-based walls
    for i in range(size):
        for j in range(size):
            # Convert grid coordinates to Perlin noise space according to the specified scale
            noise_value = pnoise2(i / scale, j / scale, octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            # Normalize and threshold the noise value to decide if a wall should be placed
            if noise_value > 0.5 and grid[i, j] == 0:  # Assuming normalized noise values; adjust threshold as needed
                grid[i, j] = -1

    return grid


import numpy as np


def generate_initial_state_simplex_noise(size, goal_number, scale, octaves, persistence, lacunarity):
    grid = np.zeros((size, size))

    # Place goal states randomly
    for _ in range(goal_number):
        while True:
            x, y = np.random.randint(size, size=2)
            if grid[x, y] == 0:
                grid[x, y] = 1
                break

    # Generate Simplex noise-based walls
    for i in range(size):
        for j in range(size):
            # Convert grid coordinates to Simplex noise space according to the specified scale
            noise_value = snoise2(i / scale, j / scale, octaves=octaves, persistence=persistence,
                                        lacunarity=lacunarity)
            # Normalize and threshold the noise value to decide if a wall should be placed
            if noise_value > 0.5 and grid[i, j] == 0:  # Adjust threshold as needed
                grid[i, j] = -1

    return grid



def generate_null_policy_fixed(initial_value_array):
    # Determine the shape from the initial value array
    policy_shape = initial_value_array.shape
    # Initialize an empty array for the policy
    null_policy = np.empty(policy_shape, dtype=object)

    # Populate the policy array with (0, 0) actions
    for i in range(policy_shape[0]):
        for j in range(policy_shape[1]):
            null_policy[i, j] = (0, 0)

    return null_policy

