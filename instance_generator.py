from random import uniform

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