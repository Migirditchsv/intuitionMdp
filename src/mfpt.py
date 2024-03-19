import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, diags
from src.math_utils import add_tuples
from src.policy_value_iteration import get_next_states
from src.visualization import plot_transition_matrix, plot_mu_matrix


def construct_transition_matrix(policy_grid, world_model):
    # Copy in action space from world model
    action_space = world_model.get_action_space()
    N = len(policy_grid)  # Size of the grid
    M = N * N
    transition_matrix = np.zeros((M, M))

    for i in range(N):
        for j in range(N):
            next_states = get_next_states((i, j), action_space, world_model)
            for new_state, prob in next_states.items():
                # Convert 2D indices to 1D index
                old_state_1d = i * N + j
                new_state_1d = new_state[0] * N + new_state[1]
                transition_matrix[old_state_1d][new_state_1d] = prob

    return transition_matrix


def compute_mfpt(policy_grid, world_model):
    # Construct the transition matrix
    transition_matrix = construct_transition_matrix(policy_grid, world_model)
    if is_singular(transition_matrix):
        print("WARNING: The transition matrix is singular, and the mean first passage time cannot be computed."
              "\n Adding a small constant to the diagonal elements to remove singularity.")
        transition_matrix = remove_singularity(transition_matrix)
    # Convert T to a sparse matrix in CSR format
    # T_sparse = csr_matrix(transition_matrix)

    # expected hitting time mu = 1 + P mu , where 1 is a vector of ones
    # -> (I-P)^(-1) * 1 = mu

    # Compute (I-P)^(-1)
    I = np.eye(transition_matrix.shape[0])
    T_inv = np.linalg.inv(I - transition_matrix)

    # Compute mu
    mu = np.dot(T_inv, np.ones(transition_matrix.shape[0]))

    N = int(np.sqrt(len(mu)))
    # Assert that N is the same size as the policy grid
    assert N == len(policy_grid)
    # Reshape mu into an NxN matrix
    mu_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            mu_matrix[i][j] = mu[i * N + j]

    return mu_matrix, transition_matrix


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
