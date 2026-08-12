from random import uniform
import numpy as np


def generate_random_symmetric_nonneg_matrix(n,
                                     low=0.0,
                                     high=1.0,
                                     integer=False,
                                     seed=None,
                                     dtype=float):
    """
    Generates a symmetric n x n matrix D with:
      - D[i,i] = 0
      - D[i,j] = D[j,i] >= 0 for i != j

    Parameters:
      - n (int): matrix size
      - low (float): minimum random value
      - high (float): maximum random value
      - integer (bool): whether to generate ints instead of floats
      - seed (int|None): RNG seed for reproducibility
      - dtype: final numpy dtype

    Returns:
      - D (numpy.ndarray): symmetric non-negative matrix
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if low < 0 or high < 0:
        raise ValueError("low and high must be non-negative.")
    if high < low:
        raise ValueError("high must be >= low.")

    rng = np.random.default_rng(seed)

    # generate full upper triangle
    if integer:
        # randint is upper-exclusive, so use high+1
        upper = rng.integers(low, high + 1, size=(n, n))
    else:
        upper = rng.random((n, n)) * (high - low) + low

    # keep only upper triangle (excluding diagonal)
    D = np.triu(upper, k=1)

    # reflect to make it symmetric
    D = D + D.T

    # ensure diagonal = 0
    np.fill_diagonal(D, 0)

    return D.astype(dtype)

def perturb_additive_matrix(D, weights):
    """
    Add symmetric random noise to a leaf distance matrix D.

    Parameters:
        D       : numpy array (N x N), symmetric
        weights : list or dict of edge weights

    Returns:
        D_new   : new perturbed symmetric numpy array
    """

    # Convert to array
    D = np.asarray(D, dtype=float)
    N = D.shape[0]

    # Extract min weight
    if isinstance(weights, dict):
        w_min = min(weights.values())
    else:
        w_min = min(weights)

    epsilon = w_min / 2.1

    # Create copy
    D_new = D.copy()

    # Apply noise symmetrically
    for i in range(N):
        for j in range(i + 1, N):
            noise = uniform(-epsilon, epsilon)
            Dij = D_new[i, j] + noise
            D_new[i, j] = Dij
            D_new[j, i] = Dij

    return D_new






