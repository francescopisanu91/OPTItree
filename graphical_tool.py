import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def draw_graph_circular(G, S1, S2, known_weights, unknown_edges):
    """
    Draw the graph G with S1 and S2 placed on two circles.
    Edges with known weights are colored using a colormap.
    Unknown edges are drawn in faint gray.
    """

    # --------------------------------------------
    # 1. Node positions on circles
    # --------------------------------------------
    def circle_positions(nodes, radius):
        k = len(nodes)
        return {
            node: (
                radius * np.cos(2 * np.pi * idx / k),
                radius * np.sin(2 * np.pi * idx / k)
            )
            for idx, node in enumerate(nodes)
        }

    pos_S1 = circle_positions(S1, radius=2.0)
    pos_S2 = circle_positions(S2, radius=1.2)
    positions = {**pos_S1, **pos_S2}

    # --------------------------------------------
    # 2. Colormap for known weights
    # --------------------------------------------
    if known_weights:
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmin = min(known_weights.values())
        cmax = max(known_weights.values())
        normalize = lambda w: (w - cmin) / (cmax - cmin + 1e-9)
    else:
        normalize = lambda w: 0.5

    # Unknown edges as undirected set
    unknown_set = set(tuple(sorted(e)) for e in unknown_edges)

    # --------------------------------------------
    # 3. Begin drawing
    # --------------------------------------------
    plt.figure(figsize=(8, 8))

    # Draw nodes
    plt.scatter([positions[i][0] for i in S1],
                [positions[i][1] for i in S1],
                s=260, color="skyblue", label="S1")

    plt.scatter([positions[i][0] for i in S2],
                [positions[i][1] for i in S2],
                s=260, color="orange", label="S2")

    # Labels
    for n, (x, y) in positions.items():
        plt.text(x, y, str(n), ha="center", va="center", fontsize=11)

    # --------------------------------------------
    # 4. Draw edges manually (like your old function)
    # --------------------------------------------
    for (u, v) in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]

        e_sorted = tuple(sorted((u, v)))

        # Known weight?
        if e_sorted in known_weights:
            w = known_weights[e_sorted]

            plt.plot([x1, x2], [y1, y2],
                     color=cmap(normalize(w)),
                     alpha=0.35,
                     linewidth=1.8)

        else:
            # Unknown weight
            plt.plot([x1, x2], [y1, y2],
                     color="gray",
                     alpha=0.15,
                     linewidth=1.0)

    plt.title("Graph instance (known weights colored, unknown edges in gray)", fontsize=14)
    plt.axis("equal")
    plt.axis("off")
    plt.legend()
    plt.show()



def draw_graph_with_solution(V, E, costs, S1, S2, selected_edges):
    """
    Draw the instance graph together with the solution (thick red trasparent edges)
    """

    # ----- Circular configuration for the vertices -----
    def circle_positions(nodes, radius):
        k = len(nodes)
        return {
            node: (
                radius * np.cos(2 * np.pi * idx / k),
                radius * np.sin(2 * np.pi * idx / k)
            )
            for idx, node in enumerate(nodes)
        }

    pos_S1 = circle_positions(S1, radius=2.0)
    pos_S2 = circle_positions(S2, radius=1.0)
    positions = {**pos_S1, **pos_S2}

    # ----- Gradient coloring of the edges based on the weight -----
    cmap = sns.color_palette("viridis", as_cmap=True)
    cmin, cmax = min(costs), max(costs)
    normalize = lambda x: (x - cmin) / (cmax - cmin + 1e-9)

    plt.figure(figsize=(8, 8))

    # ----- Draw the vertices -----
    plt.scatter([positions[i][0] for i in S1], [positions[i][1] for i in S1],
                s=200, color="skyblue", label="S1")
    plt.scatter([positions[i][0] for i in S2], [positions[i][1] for i in S2],
                s=200, color="orange", label="S2")

    # Vertex labels
    for n, (x, y) in positions.items():
        plt.text(x, y, str(n), ha="center", va="center", fontsize=12)

    # ----- Draw the instance graph edges -----
    for (e, cost) in zip(E, costs):
        u, v = e
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        plt.plot([x1, x2], [y1, y2],
                 color=cmap(normalize(cost)),
                 alpha=0.3,          
                 linewidth=1.5)      

    # ----- Draw the thick red edges to highlight the optimal solution -----
    first_label = True
    for (u, v) in selected_edges:
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        plt.plot([x1, x2], [y1, y2],
                 color="red",
                 linewidth=6,        
                 alpha=0.6,          
                 label="Optimal UBT" if first_label else "")
        first_label = False

    plt.title("Solution to the problem", fontsize=14)
    plt.axis("equal")
    plt.axis("off")
    plt.legend()
    plt.show()