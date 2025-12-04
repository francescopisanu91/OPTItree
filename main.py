import graphical_tool
import instance_generator
import instance_converter
import minimum_weighted_UBT_finder
import numpy as np 

# 1. Generate a random matrix D
D = instance_generator.generate_symmetric_nonneg_matrix(5)
print(D)

# 2. Generate and draw the weighted complete graph associated with D
G, S1, S2, weigths = instance_generator.build_complete_graph_from_D(D)
print(G)
graphical_tool.draw_graph_instance(G, S1, S2, weigths )

# 3. Generate and draw the unweighted star graph H on which the UBT lies
H, S1, S2 = instance_generator.build_star_graph_from_D(D)
print(G)
graphical_tool.draw_graph_instance(H, S1, S2 )

# 4. Generate the line graph of G
LG, edge_list, edge_to_idx = instance_converter.line_graph(G)
LG_nodes = list(LG.nodes())   # Each node = an edge (u,v) in G

# 5. Random weights for ALL LG nodes
LG_weights = np.random.rand(len(LG_nodes))

# 6. Mask: True = keep (unknown), False = drop (known)
mask = []
for (u, v) in LG_nodes:
    if (u in S1) and (v in S1):
        mask.append(False)    # known edge (inside S1)
    else:
        mask.append(True)     # unknown edge
mask = np.array(mask)

# 7. Filtered weights (only unknown ones)
LG_weights_filtered = LG_weights[mask]

# -------------------------------
# BUILD unknown_edges and known_weights
# -------------------------------

# unknown edges = all LG nodes with mask=True
unknown_edges = [LG_nodes[i] for i in range(len(LG_nodes)) if mask[i]]


print("LG nodes:", LG_nodes)
print("All weights:", LG_weights)
print("Filtered weights:", LG_weights_filtered)

graphical_tool.draw_graph_instance(G, S1, S2, unknown_edges ) ####PROBLEMMMMMMMMMMMMMMMMMMMMMMMMMMM

# # 8. Solve exact instance
# minimum_weighted_UBT_finder.ubt_solver_from_star_graph(H, S1, S2, LG_weights_filtered)