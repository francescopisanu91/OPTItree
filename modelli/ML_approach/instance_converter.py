import networkx as nx
import random
import numpy as np

# -------------------------------------------------------
# Line graph
# -------------------------------------------------------
def line_graph(G):
     LG = nx.line_graph(G)
     # map edges to indices
     edge_list = list(G.edges())
     edge_to_idx = {e:i for i,e in enumerate(edge_list)}
     return LG, edge_list, edge_to_idx


import numpy as np

def apply_mask_to_weights(LG_nodes, LG_weights, S1):
    """
    LG_nodes: list of edges (u,v) representing nodes of the line graph
    LG_weights: np.array of weights corresponding to LG_nodes
    S1: list of nodes in S1

    Returns:
        filtered_weights_dict: dict { (u,v): weight } with only edges where not both in S1
        filtered_weights_list: list of weights in the same order as filtered nodes
    """
    filtered_weights_dict = {}
    filtered_weights_list = []

    for idx, (u, v) in enumerate(LG_nodes):
        if u in S1 and v in S1:
            continue  # drop this edge entirely
        w = LG_weights[idx]
        filtered_weights_dict[(u, v)] = w
        filtered_weights_list.append(w)

    return filtered_weights_dict, filtered_weights_list

