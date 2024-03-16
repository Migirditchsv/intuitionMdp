import numpy as np
from src.math_utils import add_tuples

def construct_transition_matrix(policy_grid, action_space, stochasticity):
    # The following code is based on (Debnath 2019 paper, eq 8). It constructs the transition matrix for the given policy
    N = len(policy_grid)  # Size of the grid
    M= N*N
    transition_matrix = np.zeros((M, M))

    for i in range(N):
        for j in range(N):
            policy_action = policy_grid[i][j]
            for action in action_space.values():
                #if np.array_equal(policy_action, action):
                    new_state = add_tuples((i, j), action)
                    # Ensure the new state is within bounds
                    if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                        # Convert 2D indices to 1D index
                        old_state_1d = i * N + j
                        new_state_1d = new_state[0] * N + new_state[1]

                        # Update the transition matrix: If the action is stationary and NOT stochastic, the value is p_ii - 1 =  1 - stochasticity -1 = stochasticity
                        if old_state_1d == new_state_1d and np.array_equal(action, policy_action):
                            transition_matrix[old_state_1d][new_state_1d] += stochasticity

                        # If the policy action is stationary, and the action IS stoch, the value is stochasticity - 1
                        elif old_state_1d == new_state_1d and not np.array_equal(action, policy_action):
                            transition_matrix[old_state_1d][new_state_1d] += stochasticity -1
                        # If the policy isn't stationary, and the action is the policy action, the value is p = 1 - stochasticity
                        elif old_state_1d != new_state_1d and np.array_equal(action, policy_action):
                            transition_matrix[old_state_1d][new_state_1d] += 1 - stochasticity

                        elif old_state_1d != new_state_1d and not np.array_equal(action, policy_action):
                            transition_matrix[old_state_1d][new_state_1d] += stochasticity
                        else:
                            raise IndexError("ERROR: Weird transition action taken from state" + str((i, j)) + " to state " +
                                  str(new_state) + " with action " + str(action) + " and policy action " +
                                  str(policy_action))
                            exit(1)


    # Subtract diagonal elements from 1
    for i in range(N*N):
        transition_matrix[i][i] = 1 - transition_matrix[i][i]

    return transition_matrix

