import networkx as nx
import random

# -------------------------------------------------------
# Line graph
# -------------------------------------------------------
def line_graph(G):
     LG = nx.line_graph(G)
     # map edges to indices
     edge_list = list(G.edges())
     edge_to_idx = {e:i for i,e in enumerate(edge_list)}
     return LG, edge_list, edge_to_idx