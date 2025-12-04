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

import networkx as nx

# def build_complete_graph_from_D(D):
#     """
#     Build the graph G used as input for the line graph construction.
#     Compatible with NetworkX line_graph().

#     Input:
#         D : numpy array (n x n), symmetric, non-negative, zero diagonal
    
#     Output:
#         G : networkx.Graph
#         known_weights : dict { (i,j) : d_ij }
#         unknown_edges : list of (i,j)
#     """

#     # 1. Extract n
#     D = np.asarray(D)
#     if D.shape[0] != D.shape[1]:
#         raise ValueError("D must be a square matrix.")
#     n = D.shape[0]

#     # 2. Generate S1 and S2
#     S1 = list(range(1, n + 1))                 # size n
#     S2 = list(range(n + 1, 2*n - 1))           # size n-2

#     # Full vertex set
#     V = S1 + S2

#     # 3. Create a complete graph on V
#     G = nx.Graph()
#     G.add_nodes_from(V)

#     for i in range(len(V)):
#         for j in range(i + 1, len(V)):
#             G.add_edge(V[i], V[j])

#     known_weights = {}   # edges inside S1 (use D)
#     unknown_edges = []   # all other edges

#     # 4 + 5. Assign weights
#     for u, v in G.edges():
#         if u in S1 and v in S1:
#             # indices for D are (u-1, v-1)
#             w = float(D[u-1, v-1])
#             G[u][v]['weight'] = w
#             known_weights[(u, v)] = w
#         else:
#             # unknown weight (to be predicted on the line graph)
#             G[u][v]['weight'] = 0.0
#             unknown_edges.append((u, v))

#     return G, S1, S2, known_weights, unknown_edges

def build_complete_graph_from_D(D):
    """
    Build the graph G used as input for the line graph construction.
    Compatible with NetworkX line_graph().

    Input:
        D : numpy array (n x n), symmetric, non-negative, zero diagonal
    
    Output:
        G : networkx.Graph
        S1 : list of first vertex subset
        S2 : list of second vertex subset
        edge_weights : dict { (i,j) : weight } in lexicographic order
    """

    import networkx as nx
    import numpy as np

    # 1. Extract n
    D = np.asarray(D)
    if D.shape[0] != D.shape[1]:
        raise ValueError("D must be a square matrix.")
    n = D.shape[0]

    # 2. Generate S1 and S2
    S1 = list(range(1, n + 1))                 # size n
    S2 = list(range(n + 1, 2*n - 1))           # size n-2

    # Full vertex set
    V = S1 + S2

    # 3. Create a complete graph on V
    G = nx.Graph()
    G.add_nodes_from(V)

    edge_weights = {}

    for i in range(len(V)):
        for j in range(i + 1, len(V)):
            u, v = V[i], V[j]
            G.add_edge(u, v)
            e_sorted = tuple(sorted((u, v)))

            if u in S1 and v in S1:
                # indices for D are (u-1, v-1)
                w = float(D[u-1, v-1])
            else:
                # unknown weight → initialize to 0
                w = 0.0

            G[u][v]['weight'] = w
            edge_weights[e_sorted] = w

    return G, S1, S2, edge_weights



def build_star_graph_from_D(D):
    """
    Build the star graph G used as input for the NN construction.
    - S1 = {1, ..., n}
    - S2 = {n+1, ..., 2n-2}
    - every i in S1 is universal towards S2
    - S2 is a clique
    Input:
        D : numpy array (usato solo per ricavare n)

    Output:
        G  : networkx.Graph
        S1 : vertex list
        S2 : vertex list
    """

    D = np.asarray(D)
    if D.shape[0] != D.shape[1]:
        raise ValueError("D deve essere una matrice quadrata")

    n = D.shape[0]

    # --- Sets definition ---
    S1 = list(range(1, n + 1))                 # n nodi
    S2 = list(range(n + 1, 2*n - 1))           # n-2 nodi

    G = nx.Graph()
    G.add_nodes_from(S1)
    G.add_nodes_from(S2)

    # --- S1 to S2 edges ---
    for u in S1:
        for v in S2:
            G.add_edge(u, v, weight=1.0)

    # --- S2 induces a clique ---
    for i in range(len(S2)):
        for j in range(i + 1, len(S2)):
            u = S2[i]
            v = S2[j]
            G.add_edge(u, v, weight=1.0)


    return G, S1, S2


