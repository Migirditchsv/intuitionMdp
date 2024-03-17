import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, diags
from src.math_utils import add_tuples
from src.visualization import plot_transition_matrix


def construct_transition_matrix(policy_grid, action_space, stochasticity):
    N = len(policy_grid)  # Size of the grid
    M = N * N
    transition_matrix = np.zeros((M, M))

    for i in range(N):
        for j in range(N):
            policy_action = policy_grid[i][j]
            for action in action_space.values():
                new_state = add_tuples((i, j), action)
                # Ensure the new state is within bounds
                if 0 <= new_state[0] < N and 0 <= new_state[1] < N:
                    # Convert 2D indices to 1D index
                    old_state_1d = i * N + j
                    new_state_1d = new_state[0] * N + new_state[1]

                    # Update the transition matrix based on Debnath et al. (2019) eq (8)
                    if old_state_1d == new_state_1d: # If T_ii
                        if np.array_equal(action, policy_action):
                            transition_matrix[old_state_1d][new_state_1d] += -stochasticity
                        else:
                            transition_matrix[old_state_1d][new_state_1d] += stochasticity
                    else:
                        if np.array_equal(action, policy_action):
                            transition_matrix[old_state_1d][new_state_1d] += 1 - stochasticity
                        else:
                            transition_matrix[old_state_1d][new_state_1d] += stochasticity

    # Subtract diagonal elements from 1
    for i in range(M):
        transition_matrix[i][i] = 1 - transition_matrix[i][i]

    return transition_matrix


def compute_mfpt(policy_grid, action_space, stochasticity):

    # Construct the transition matrix
    transition_matrix = construct_transition_matrix(policy_grid, action_space, stochasticity)

    if is_singular(transition_matrix):
        print("WARNING: The transition matrix is singular, and the mean first passage time cannot be computed."
              "\n Adding a small constant to the diagonal elements to remove singularity.")
        transition_matrix = remove_singularity(transition_matrix)
    # Convert T to a sparse matrix in CSR format
    #T_sparse = csr_matrix(transition_matrix)

    # Simply invert Trans matrix
    T_inv = np.linalg.inv(transition_matrix)

    # Create a negative vector of ones
    id_vector = -np.ones(transition_matrix.shape[0])

    # Solve the system
    mu = T_inv @ id_vector

    N = int(np.sqrt(len(mu)))
    # Assert that N is the same size as the policy grid
    assert N == len(policy_grid)
    # Reshape mu into an NxN matrix
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