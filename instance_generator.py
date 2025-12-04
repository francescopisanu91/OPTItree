from random import uniform
import numpy as np


def generate_symmetric_nonneg_matrix(n,
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


def weighted_graph_generator(n, low=1.0, high=10.0):
    """
    Generate a graph such that:
    - S1 = {1, ..., n}
    - S2 = {n+1, ..., 2n-2}
    - every i in S1 is universal towards S2
    - S2 is a clique
    - all the edges are weighted uniformly randomly in [low, high]

    Ritorna:
    - V: vertex set S1US2
    - E: edge list (u, v)
    - costs: wieght list sorted with respect to E
    """
    # Vertex sets definition
    S1 = list(range(1, n+1))
    S2 = list(range(n+1, 2*n-1))   # n-2 vertex

    V = S1 + S2
    E = []

    # Edges from S1 towards S2
    for u in S1:
        for v in S2:
            E.append((u, v))

    # S2 clique (symmetries are broken)
    for i in range(len(S2)):
        for j in range(i+1, len(S2)):
            E.append((S2[i], S2[j]))

    # Random weights generator
    costs = [round(uniform(low, high), 2) for _ in E]

    return V, E, costs


