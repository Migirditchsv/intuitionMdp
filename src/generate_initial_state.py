import random
import numpy as np
from noise import snoise2

def generate_simple_initial_value(size, random_seed=None):

    # Set the random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

    # Create a size x size array filled with zeros
    array = np.zeros((size, size))

    # Calculate the center of the array
    center = size // 2

    # Set the center cell to 1 (goal state)
    array[center][center] = 1

    return array

def generate_initial_values(size, goal_number, density, wall_clustering, random_seed=None):
    # Set the random seed if provided
    if random_seed is not None:
        random.seed(random_seed)

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

def generate_initial_values_simplex(size, goal_number, scale, octaves, persistence, lacunarity, threshold):
    # Initialize the map with zeros
    map_array = np.zeros((size, size))

    # Place goals
    goals_placed = 0
    while goals_placed < goal_number:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if map_array[x, y] == 0:  # Ensure not placing a goal on top of a wall or another goal
            map_array[x, y] = 1.0
            goals_placed += 1

    # Iterate over each cell in the map
    for i in range(size):
        for j in range(size):
            # Generate simplex noise for the cell
            noise_value = snoise2(i / scale,
                                  j / scale,
                                  octaves=octaves,
                                  persistence=persistence,
                                  lacunarity=lacunarity)

            # If the noise value is above the threshold, mark the cell as a wall
            # Ensure not placing a wall on a goal
            if noise_value > threshold and map_array[i, j] != 1.0:
                map_array[i][j] = -1.0

    return map_array

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

