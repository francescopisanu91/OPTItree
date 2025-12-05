import networkx as nx
import numpy as np
from random import uniform

def random_tree_with_n_leaves(n, min_weight=0.0, max_weight=5.0):
    """
    Generate a random tree with exactly n leaves.
    Each edge gets a random weight in [min_weight, max_weight].
    
    Returns:
        T        : networkx.Graph (tree)
        leaves   : list of leaf nodes
        weights  : dict  {(u,v): w} with sorted tuple keys
    """
    # Start by creating a random labeled tree using Prüfer sequence
    # Prüfer sequence of length n-2 generates a tree with n nodes
    T = nx.random_labeled_tree(n)

    # Identify leaves (degree 1 nodes)
    leaves = [v for v in T.nodes() if T.degree(v) == 1]

    # Assign random weights to edges
    weights = {}
    for (u, v) in T.edges():
        w = uniform(min_weight, max_weight)
        weights[tuple(sorted((u, v)))] = w
        T[u][v]['weight'] = w  # store weight inside the graph too

    return T, leaves, weights

import networkx as nx

def tree_cubifier(T, weights):
    """
Normalize a tree T in two phases:

1) For every node with degree > 3:
   the node is replaced by an edge f = (vA, vB) connecting two new nodes vA and vB,
   with weight 0. The neighbors of the original node are split between vA and vB.

2) For every node with degree = 2:
   the node is suppressed and its two neighbors are directly connected.
   The new edge receives weight = sum of the two removed edges.

Parameters:
    T        : networkx.Graph, a tree
    weights  : dict { (u, v) : w } with sorted tuples as keys

Returns:
    T'       : the modified tree
    weights' : the updated dictionary of edge weights
"""

    T = T.copy()
    weights = {tuple(sorted(k)): v for k, v in weights.items()}

   # ------------------------------------------------------------
    # STEP 1 – Replace nodes with degree > 3 (iteratively)
    # ------------------------------------------------------------
    next_id = max(T.nodes()) + 1

    while True:
        # find a vertex with deg > 3
        big_nodes = [v for v in T.nodes() if T.degree(v) > 3]
        if not big_nodes:
            break  

        v = big_nodes[0]  

        neigh = list(T.neighbors(v))
        deg = len(neigh)

        mid = 2
        groupA = neigh[:mid]
        groupB = neigh[mid:]
        
        # crea due nuovi nodi
        vA = next_id; next_id += 1
        vB = next_id; next_id += 1
        T.add_node(vA)
        T.add_node(vB)

        # aggiungi arco f = (vA, vB)
        f = tuple(sorted((vA, vB)))
        T.add_edge(*f)
        weights[f] = 0.0

        # riattacca i vicini
        for u in groupA:
            old_key = tuple(sorted((u, v)))
            w = weights.get(old_key, 0.0)
            new_key = tuple(sorted((u, vA)))

            T.add_edge(u, vA)
            weights[new_key] = w

        for u in groupB:
            old_key = tuple(sorted((u, v)))
            w = weights.get(old_key, 0.0)
            new_key = tuple(sorted((u, vB)))

            T.add_edge(u, vB)
            weights[new_key] = w

        # elimina v e i vecchi archi
        for u in neigh:
            old_key = tuple(sorted((u, v)))
            if old_key in weights:
                del weights[old_key]

        T.remove_node(v)


    # ------------------------------------------------------------
    # STEP 2 – Soppressione nodi con grado = 2
    # ------------------------------------------------------------
    changed = True
    while changed:
        changed = False
        nodes = list(T.nodes())

        for v in nodes:
            if v not in T:
                continue
            if T.degree(v) != 2:
                continue

            # i due vicini
            n1, n2 = list(T.neighbors(v))

            # pesi originali degli archi
            e1 = tuple(sorted((v, n1)))
            e2 = tuple(sorted((v, n2)))
            w1 = weights.get(e1, 0.0)
            w2 = weights.get(e2, 0.0)

            # nuovo arco tra n1 e n2
            e_new = tuple(sorted((n1, n2)))
            w_new = w1 + w2

            T.add_edge(*e_new)
            weights[e_new] = w_new

            # rimuovi vecchi archi
            if e1 in weights:
                del weights[e1]
            if e2 in weights:
                del weights[e2]
            T.remove_node(v)

            changed = True
            break

    return T, weights

def relabel_tree_leaves_first(T, weights):
    """
    Relabels a tree so that:
      - Leaves are labeled 1..ℓ
      - Internal nodes are labeled ℓ+1..n

    Params:
        T        : networkx.Graph, a tree
        weights  : dict {(u,v): w} with sorted tuple keys

    Returns:
        T2        : relabeled tree
        leaves2   : list of leaf labels (1..ℓ)
        weights2  : updated weight dictionary
    """
    # --- Identify old leaves and internal nodes ---
    leaves = [v for v in T.nodes() if T.degree(v) == 1]
    internals = [v for v in T.nodes() if T.degree(v) > 1]

    ℓ = len(leaves)

    # --- Build new labeling ---
    new_labels = {}
    next_leaf_label = 1
    next_internal_label = ℓ + 1

    # assign leaf labels
    for v in leaves:
        new_labels[v] = next_leaf_label
        next_leaf_label += 1

    # assign internal labels
    for v in internals:
        new_labels[v] = next_internal_label
        next_internal_label += 1

    # --- Apply relabeling to graph ---
    T2 = nx.relabel_nodes(T, new_labels, copy=True)

    # --- Rebuild weights with new labels ---
    weights2 = {}
    for (u, v), w in weights.items():
        u2 = new_labels[u]
        v2 = new_labels[v]
        key = tuple(sorted((u2, v2)))
        weights2[key] = w
        T2[u2][v2]['weight'] = w

    # --- Leaves are now exactly 1..ℓ ---
    leaves2 = list(range(1, ℓ + 1))

    return T2, leaves2, weights2

def additive_matrix(T, leaves, weights):
    """
    Compute pairwise leaf distance matrix D for a weighted tree.

    Parameters:
        T       : networkx.Graph (tree)
        leaves  : list of leaf node labels
        weights : dict { (u,v): w } with sorted (u,v)

    Returns:
        D       : numpy array (L x L) where D[i,j] = distance between leaves[i] and leaves[j]
    """

    L = len(leaves)
    D = np.zeros((L, L))

    # Precompute weight lookup for speed
    def w(u, v):
        return weights.get(tuple(sorted((u, v)))), 0.0

    # For each unordered pair i<j
    for i in range(L):
        for j in range(i + 1, L):
            leaf_i = leaves[i]
            leaf_j = leaves[j]

            # path is unique because T is a tree
            path = nx.shortest_path(T, leaf_i, leaf_j)

            # sum weights on consecutive edges
            dist = 0.0
            for k in range(len(path) - 1):
                e = tuple(sorted((path[k], path[k+1])))
                dist += weights[e]

            D[i, j] = D[j, i] = dist

    return D