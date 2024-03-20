import numpy as np

from src.math_utils import is_out_of_bounds


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


def value_iteration_step(world_model, value_grid, policy_grid, gamma=0.9, update_states=None):
    # Pull in a copy of the state space
    state_space = world_model.get_world_map()

    # Ensure world_model.state_space, value_grid, and policy_grid are all the same size
    assert len(world_model.get_world_map()) == len(value_grid) == len(policy_grid)

    # Initialize grids for values and policies, and max_delta_value
    N = len(value_grid)  # Size of the grid
    new_value_grid = value_grid.copy()

    max_delta_value = 0  # To track the maximum change in value

    # Iterate over each state in the grid if
    if update_states is None:
        update_states = [(i, j) for i in range(N) for j in range(N)]
    for state in update_states:
        if new_value_grid[state] == world_model.goal_value:  # Goal state
            pass  # continue
        elif new_value_grid[state] == world_model.wall_value:  # Wall
            pass  # continue
        value = 0.0
        next_states = get_next_states(state, policy_grid[state], world_model)
        for new_state, prob in next_states.items():
            # Ensure the new state is within bounds
            if is_out_of_bounds(new_state, state_space):  # Out of bounds
                new_state = state  # Deterministically stay in bounds
            reward = get_reward(state, policy_grid[state], new_state, world_model)
            value += prob * (reward + gamma * value_grid[new_state])  # Bellman equation
            # if value > 1:
            #     print("WARNING: Value ", value, " is greater than 1 during value iteration transition from "
            #           , state, " to ", new_state, " under action ", policy_grid[state])
        # Update the value grid and max_delta_value
        delta_value = abs(value - value_grid[state])
        if delta_value > max_delta_value:
            max_delta_value = delta_value
        new_value_grid[state] = value
    return new_value_grid, max_delta_value


def policy_iteration_step(world_model, value_grid, policy_grid, gamma, update_states=None):
    # Pull in a copy of the state space
    state_space = world_model.get_world_map()
    # Ensure world_model.state_space, value_grid, and policy_grid are all the same size
    assert len(state_space) == len(value_grid) == len(policy_grid)

    # Initialize grids for values and policies, and max_delta_value
    N = len(value_grid)  # Size of the grid
    new_policy_grid = policy_grid.copy()
    policy_unstable = True  # Track if the policy has changed

    # Iterate over each state in the grid if
    if update_states is None:
        update_states = [(i, j) for i in range(N) for j in range(N)]
    for state in update_states:  # Update the policy for each state
        if state_space[state] == world_model.goal_value:  # Goal state
            pass  # continue
        elif state_space[state] == world_model.wall_value:  # Wall
            pass  # continue
        max_value = -float('inf')  # Refresh max value for the state
        best_action = (0, 0)  # Default to stationary action
        for action in world_model.action_space.values():  # Check the value of each action
            value = 0.0  # Initialize value for the action
            next_states = get_next_states(state, action, world_model)
            for new_state, prob in next_states.items():  # Check value contribution from each possible next state under the action
                # Ensure the new state is within bounds
                if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                    reward = get_reward(state, policy_grid[state], new_state, world_model)
                    value += prob * (reward + gamma * value_grid[new_state])  # Bellman equation
                else:
                    # State is out of bounds
                    print("WARNING: State is out of bounds during value iteration transition from "
                          , state, " to ", new_state, " under action ", policy_grid[state])
                    exit(1)
                if value > max_value:
                    max_value = value
                    best_action = action
        # Check for policy improvement
        if best_action != policy_grid[state]:
            policy_unstable = True
            new_policy_grid[state] = best_action
    return new_policy_grid, policy_unstable


def policy_iteration_mfpt_step(world_model, value_grid, policy_grid, mfpt_array, update_states=None):
    # Get a copy of the state space
    state_space = world_model.get_world_map()
    # Ensure world_model.state_space, value_grid, and policy_grid are all the same size
    assert len(state_space) == len(value_grid) == len(policy_grid)

    # Initialize grids for values and policies, and max_delta_value
    N = len(value_grid)  # Size of the grid
    new_policy_grid = policy_grid.copy()

    # Iterate over each state in the grid if
    if update_states is None:
        update_states = [(i, j) for i in range(N) for j in range(N)]
    for state in update_states:  # Update the policy for each state
        if state_space[state] == world_model.goal_value:  # Goal state
            pass  # continue
        elif state_space[state] == world_model.wall_value:  # Wall
            pass  # continue
        min_mfpt_value = float('inf')
        best_action = (0, 0)  # Default to stationary action
        for action in world_model.action_space.values():  # Check the value of each action
            mfpt_value = 0.0  # Initialize value for the action
            next_states = get_next_states(state, action, world_model)
            for new_state, prob in next_states.items():  # Check expected MFPT value for each possible next state
                # under the action
                # Ensure the new state is within bounds
                if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                    mfpt_value += prob * mfpt_array[new_state]  # Contribution to the expected MFPT value of the action
                else:
                    # State is out of bounds
                    print("WARNING: State is out of bounds during value iteration transition from "
                          , state, " to ", new_state, " under action ", policy_grid[state])
                    exit(1)
                if mfpt_value < min_mfpt_value:
                    min_mfpt_value = mfpt_value
                    best_action = action
        new_policy_grid[state] = best_action
        # Don't check for policy stability here, as the MFPT policy is not expected to be stable
    return new_policy_grid


def get_next_states(state, policy_action, world_model):
    next_states = {}
    action_space = world_model.get_action_space()
    num_actions = len(action_space)
    world_map = world_model.get_world_map()
    N = len(world_map)

    # If state is goal, return the same state with 100% probability
    if world_map[state] == world_model.goal_value:
        next_states[state] = 1.0
        return next_states

    for action_name, action in action_space.items():
        new_state = (state[0] + action[0], state[1] + action[1])
        # Check if the new state is out of bounds
        if new_state[0] < 0 or new_state[0] >= N or new_state[1] < 0 or new_state[1] >= N:
            new_state = state  # Stay in place if the action would take you out of bounds
        if action == policy_action:
            # If the new_state key does not exist, create it with a value of 1 - stochasticity
            next_states[new_state] = next_states.get(new_state, 0) + 1.0 - world_model.stochasticity
        else:
            next_states[new_state] = next_states.get(new_state, 0) + world_model.stochasticity / (num_actions - 1)

    # Normalize probabilities
    total = sum(next_states.values())
    if total == 0:
        return {k: 1 / len(next_states) for k in next_states.keys()}
    for state, prob in next_states.items():
        next_states[state] = prob / total

    return next_states


# def get_next_states(state, policy_action, world_model):
#     next_states = {}
#     state_space = world_model.get_world_map()
#     # If state is goal, return the same state with 100% probability
#     if state_space[state] == world_model.goal_value:
#         next_states[state] = 1.0
#         return next_states
#     for action_name, action in world_model.get_action_space().items():
#         new_state = (state[0] + action[0], state[1] + action[1])
#         if is_out_of_bounds(new_state, state_space):  # Any action that would take you OoB makes you stay in place
#             new_state = state
#         if action == policy_action:
#             next_states[new_state] = 1.0 - world_model.stochasticity
#         else:  # A stochastic action has been taken
#             # Edge detection
#             vertical_edge = (state[0] == 0 or state[0] == len(state_space) )
#             horizontal_edge = (state[1] == 0 or state[1] == len(state_space) )
#             if vertical_edge and horizontal_edge: # Corner -> 3 = 3 moves + stationary - 1 for policy
#                 next_states[new_state] = world_model.stochasticity / 3
#             elif vertical_edge or horizontal_edge: # Edge -> 5 = 5 moves + stationary - 1 for policy
#                 next_states[new_state] = world_model.stochasticity / 5
#             else: # Middle -> 8 moves + stationary - 1 for policy
#                 next_states[new_state] = world_model.stochasticity / 8
#     # enforce that the probabilities sum to 1
#     next_states = normalize(next_states)
#     assert_probabilities_sum_to_one(next_states)
#     return next_states

def normalize(prob_dict):
    total = sum(prob_dict.values())
    if total == 0:
        return {k: 1 / len(prob_dict) for k in prob_dict.keys()}
    return {k: v / total for k, v in prob_dict.items()}


def assert_probabilities_sum_to_one(prob_dict, epsilon=1e-6):
    total = sum(prob_dict.values())
    assert abs(total - 1) <= epsilon, f"Probabilities sum to {total}, which is not within {epsilon} of 1"


def get_reward(state, action, next_state, world_model):
    # Copy in the state space
    state_space = world_model.get_world_map()
    # Check for wall hit
    if state_space[next_state] == world_model.get_wall_value():
        return world_model.wall_reward
    # Check for goal hit
    elif state_space[next_state] == world_model.get_goal_value():
        return world_model.goal_reward
    # Check if action is stationary
    elif action == (0, 0):
        return world_model.stationary_reward
    # If none of the above conditions are met, return movement_cost
    return world_model.movement_reward
