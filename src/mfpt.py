import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, diags
from src.math_utils import add_tuples
from src.visualization import plot_transition_matrix


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

def compute_mfpt(policy_grid, action_space, stochasticity):

    # Construct the transition matrix
    transition_matrix = construct_transition_matrix(policy_grid, action_space, stochasticity)
    #plot_transition_matrix(transition_matrix)
    if is_singular(transition_matrix):
        print("WARNING: The transition matrix is singular, and the mean first passage time cannot be computed."
              "\n Adding a small constant to the diagonal elements to remove singularity.")
        transition_matrix = remove_singularity(transition_matrix)
    # Convert T to a sparse matrix in CSR format
    T_sparse = csr_matrix(transition_matrix)

    # Create a negative vector of ones
    id_vector = -np.ones(transition_matrix.shape[0])

    # Solve the system
    mu = spsolve(T_sparse, id_vector)

    # Reshape mu into an NxN matrix
    N = int(np.sqrt(len(mu)))
    mu_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mu_matrix[i][j] = mu[i * N + j]

    return mu_matrix

def is_singular(matrix):
    if np.linalg.matrix_rank(matrix) < min(matrix.shape):
        return True
    else:
        return False

def remove_singularity(mtx):
    # Convert mtx to a sparse matrix in CSR format
    mtx_sparse = csr_matrix(mtx)

    # Add a small constant to the diagonal elements
    mtx_sparse = mtx_sparse + diags([1e-10] * mtx_sparse.shape[0], 0)

    # Convert the sparse matrix back to a dense matrix
    mtx = mtx_sparse.toarray()

    return mtx