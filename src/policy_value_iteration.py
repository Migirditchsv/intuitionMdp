import numpy as np


def policy_value_iteration(initialCondition, action_space, stochasticity, movement_cost_scale):
    N = len(initialCondition)  # Size of the grid
    value_grid = np.copy(initialCondition)  # Copy the initial conditions to the value grid
    movement_cost = - movement_cost_scale / (N * N)  # Movement cost for non-stationary actions

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


def value_iteration_step(world_model, value_grid, policy_grid, update_states=None):
    # Ensure world_model.state_space, value_grid, and policy_grid are all the same size
    assert len(world_model.state_space) == len(value_grid) == len(policy_grid)

    # Initialize grids for values and policies, and max_delta_value
    N = len(value_grid)  # Size of the grid
    new_value_grid = value_grid.copy()

    # Iterate over each state in the grid if
    if update_states is None:
        update_states = [(i, j) for i in range(N) for j in range(N)]
    for state in update_states:
        if world_model.state_space[state] == world_model.goal_value:  # Goal state
            continue
        elif world_model.state_space[state] == world_model.wall_value:  # Wall
            continue
        else:
            value = 0.0
            next_states = get_next_states(state, policy_grid[state], world_model)
            for new_state, prob in next_states.items():
                # Ensure the new state is within bounds
                if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                    reward = get_reward(state, policy_grid[state], new_state, world_model)
                    value += prob * ( reward + world_model.gamma * value_grid[new_state] ) # Bellman equation
                else:
                    #State is out of bounds
                    print("WARNING: State is out of bounds during value iteration transition from "
                          , state, " to ", new_state, " under action ", policy_grid[state])
                    exit(1)
            new_value_grid[state] = value
    return new_value_grid

def policy_iteration_step(world_model, value_grid, policy_grid, update_states=None):
    # Ensure world_model.state_space, value_grid, and policy_grid are all the same size
    assert len(world_model.state_space) == len(value_grid) == len(policy_grid)

    # Initialize grids for values and policies, and max_delta_value
    N = len(value_grid)  # Size of the grid
    new_policy_grid = policy_grid.copy()

    # Iterate over each state in the grid if
    if update_states is None:
        update_states = [(i, j) for i in range(N) for j in range(N)]
    for state in update_states: # Update the policy for each state
        if world_model.state_space[state] == world_model.goal_value:  # Goal state
            continue
        elif world_model.state_space[state] == world_model.wall_value:  # Wall
            continue
        else:
            max_value = -float('inf')
            best_action = new_policy_grid[state]
            for action in world_model.action_space.values(): # Check the value of each action
                value= 0.0 # Initialize value for the action
                next_states = get_next_states(state, action, world_model)
                for new_state, prob in next_states.items(): # Check value contribution from each possible next state under the action
                    # Ensure the new state is within bounds
                    if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                        reward = get_reward(state, policy_grid[state], new_state, world_model)
                        value += prob * (reward + world_model.gamma * value_grid[new_state])  # Bellman equation
                    else:
                        # State is out of bounds
                        print("WARNING: State is out of bounds during value iteration transition from "
                              , state, " to ", new_state, " under action ", policy_grid[state])
                        exit(1)
                    if value > max_value:
                        max_value = value
                        best_action = action
            new_policy_grid[state] = best_action
    return new_policy_grid

def policy_iteration_mfpt_step(world_model, value_grid, policy_grid, mfpt_array, update_states=None):
    # Ensure world_model.state_space, value_grid, and policy_grid are all the same size
    assert len(world_model.state_space) == len(value_grid) == len(policy_grid)

    # Initialize grids for values and policies, and max_delta_value
    N = len(value_grid)  # Size of the grid
    new_policy_grid = policy_grid.copy()

    # Iterate over each state in the grid if
    if update_states is None:
        update_states = [(i, j) for i in range(N) for j in range(N)]
    for state in update_states: # Update the policy for each state
        if world_model.state_space[state] == world_model.goal_value:  # Goal state
            continue
        elif world_model.state_space[state] == world_model.wall_value:  # Wall
            continue
        else:
            min_mfpt_value = float('inf')
            best_action = new_policy_grid[state]
            for action in world_model.action_space.values(): # Check the value of each action
                mfpt_value= 0.0 # Initialize value for the action
                next_states = get_next_states(state, action, world_model)
                for new_state, prob in next_states.items(): # Check expected MFPT value for each possible next state
                    # under the action
                    # Ensure the new state is within bounds
                    if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                        mfpt_value += prob * mfpt_array[new_state] # Contribution to the expected MFPT value of the action
                    else:
                        # State is out of bounds
                        print("WARNING: State is out of bounds during value iteration transition from "
                              , state, " to ", new_state, " under action ", policy_grid[state])
                        exit(1)
                    if mfpt_value < min_mfpt_value:
                        min_mfpt_value = mfpt_value
                        best_action = action
            new_policy_grid[state] = best_action
    return new_policy_grid

def get_next_states(state, policy_action, world_model):
    next_states = {}
    for action, movement in world_model.action_space.items():
        new_state = (state[0] + movement[0], state[1] + movement[1])
        if 0 <= new_state[0] < len(world_model.state_space) and 0 <= new_state[1] < len(world_model.state_space):
            if world_model.state_space[new_state[0]][new_state[1]] == -1:  # Wall
                new_state = state
                next_states[new_state] = 1.0  # If you bounce off a wall, you stay in the same state with probability 1
            elif action == policy_action:  # Not a wall
                next_states[new_state] = 1 - world_model.stochasticity  # Take the policy action with probability
                # 1 - stochasticity
            else:  # A stochastic action has been taken
                next_states[new_state] = world_model.stochasticity
    return next_states


def get_reward(state, action, next_state, world_model):
    # Check if state and next_state are the same and action is non-stationary
    if state == next_state and action != (0, 0):
        return world_model.wall_reward

    # Check if next_state is a goal state and action is non-stationary
    if world_model.state_space[next_state[0]][next_state[1]] == 1 and action != (0, 0):
        return world_model.goal_reward

    # Check if action is stationary
    if action == (0, 0):
        return world_model.stationary_reward

    # If none of the above conditions are met, return movement_cost
    return world_model.movement_reward
