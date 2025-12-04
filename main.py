import graphical_tool
import instance_generator

D = instance_generator.generate_symmetric_nonneg_matrix(5)
G, S1, S2, known_weights, unknown_edges = instance_generator.build_graph_from_D(D)
print(D)
print(G)
graphical_tool.draw_graph_circular(G, S1, S2, known_weights, unknown_edges )