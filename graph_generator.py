import networkx as nx
import numpy as np
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