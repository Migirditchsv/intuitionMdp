import numpy as np


def policy_value_iteration(initialCondition, action_space, stochasticity, movement_cost_scale):
    N = len(initialCondition)  # Size of the grid
    value_grid = np.copy(initialCondition)  # Copy the initial conditions to the value grid
    movement_cost = movement_cost_scale / (2 * N)  # Movement cost for non-stationary actions

    # Initialize grids for values and policies, and max_delta_value
    new_value_grid = np.zeros((N, N))
    policy_grid = np.empty((N, N), dtype=object)
    max_delta_value = 0  # To track the maximum change in value

    for i in range(N):
        for j in range(N):
            if initialCondition[i][j] == 1:  # Goal state
                new_value_grid[i][j] = 1
                policy_grid[i][j] = (0, 0)  # No action for goal
            elif initialCondition[i][j] == -1:  # Wall
                new_value_grid[i][j] = -1
                policy_grid[i][j] = (0, 0)  # No action for wall
            else:
                values = []
                actions = []
                for action, (dx, dy) in action_space.items():
                    new_x, new_y = i + dx, j + dy
                    # Ensure the new state is within bounds and not a wall
                    if 0 <= new_x < N and 0 <= new_y < N and initialCondition[new_x][new_y] != -1:
                        action_cost = movement_cost if (dx, dy) != (0, 0) else 0
                        values.append(value_grid[new_x][new_y] - action_cost)
                        actions.append((dx, dy))

                # Calculate expected value considering stochasticity
                if values:
                    optimal_index = np.argmax(values)
                    optimal_value = values[optimal_index]
                    random_action_value = np.mean(values)  # Average value for random action
                    expected_value = (1 - stochasticity) * optimal_value + stochasticity * random_action_value
                    new_value_grid[i][j] = expected_value

                    # Update max_delta_value if the change in value is the largest seen so far
                    delta_value = abs(expected_value - value_grid[i][j])
                    max_delta_value = max(max_delta_value, delta_value)

                    policy_grid[i][j] = actions[optimal_index]
                else:
                    new_value_grid[i][j] = 0
                    policy_grid[i][j] = (0, 0)  # No valid actions available

    return new_value_grid, policy_grid, max_delta_value


def policy_value_iteration_ranked_partial_update(initialCondition, action_space, stochasticity, movement_cost_scale, update_indices):
    N = len(initialCondition)  # Size of the grid
    value_grid = np.copy(initialCondition)  # Copy the initial conditions to the value grid
    movement_cost = movement_cost_scale / (2 * N)  # Movement cost for non-stationary actions

    # Initialize grids for values and policies, and max_delta_value
    new_value_grid = np.zeros((N, N))
    policy_grid = np.empty((N, N), dtype=object)
    max_delta_value = 0  # To track the maximum change in value


    for index in update_indices:
        i=index[0]
        j=index[1]
        if initialCondition[i][j] == 1:  # Goal state
            new_value_grid[i][j] = 1
            policy_grid[i][j] = (0, 0)  # No action for goal
        elif initialCondition[i][j] == -1:  # Wall
            new_value_grid[i][j] = -1
            policy_grid[i][j] = (0, 0)  # No action for wall
        else:
            values = []
            actions = []
            for action, (dx, dy) in action_space.items():
                new_x, new_y = i + dx, j + dy
                # Ensure the new state is within bounds and not a wall
                if 0 <= new_x < N and 0 <= new_y < N and initialCondition[new_x][new_y] != -1:
                    action_cost = movement_cost if (dx, dy) != (0, 0) else 0
                    values.append(value_grid[new_x][new_y] - action_cost)
                    actions.append((dx, dy))

            # Calculate expected value considering stochasticity
            if values:
                optimal_index = np.argmax(values)
                optimal_value = values[optimal_index]
                random_action_value = np.mean(values)  # Average value for random action
                expected_value = (1 - stochasticity) * optimal_value + stochasticity * random_action_value
                new_value_grid[i][j] = expected_value

                # Update max_delta_value if the change in value is the largest seen so far
                delta_value = abs(expected_value - value_grid[i][j])
                max_delta_value = max(max_delta_value, delta_value)

                policy_grid[i][j] = actions[optimal_index]
            else:
                new_value_grid[i][j] = 0
                policy_grid[i][j] = (0, 0)  # No valid actions available

    return new_value_grid, policy_grid, max_delta_value


