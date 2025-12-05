import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def draw_graph_instance(G, S1, S2, weights=None, title = ''):
    """
    Draw graph G with S1 and S2 placed on two circles.

    - If weights is None → every edge is treated as weight 1
    - Any edge with weight == 0 is drawn in gray
    """

    # --------------------------------------------
    # 0. Default behaviors
    # --------------------------------------------
    if weights is None:
        # every edge has weight 1
        weights = {tuple(sorted(e)): 1.0 for e in G.edges()}

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
    # 2. Colormap for positive weights
    # --------------------------------------------
    positive_weights = [w for w in weights.values() if w > 0]
    if len(positive_weights) > 0:
        cmap = sns.color_palette("viridis", as_cmap=True)
        cmin, cmax = min(positive_weights), max(positive_weights)
        normalize = lambda w: (w - cmin) / (cmax - cmin + 1e-9)
    else:
        normalize = lambda w: 0.5  # fallback

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
    # 4. Draw edges
    # --------------------------------------------
    for (u, v) in G.edges():
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        e_sorted = tuple(sorted((u, v)))

        w = weights.get(e_sorted, 0.0)  # default to 0 if missing

        if w == 0:
            # weight zero → gray
            plt.plot([x1, x2], [y1, y2],
                     color="gray", alpha=0.3, linewidth=1.2)
        else:
            # weight > 0 → heatmap
            plt.plot([x1, x2], [y1, y2],
                     color=cmap(normalize(w)),
                     alpha=0.7, linewidth=2.0)

    plt.title(title, fontsize=14)
    plt.axis("equal")
    plt.axis("off")
    plt.legend()
    plt.show()




def draw_solution(H, S1, S2, costs, selected_edges=None):
    """
    Draw the NetworkX graph H with circular layout for S1 and S2.
    Edges are colored based on the provided costs list.
    
    Inputs:
        H : networkx.Graph
        S1, S2 : lists of nodes
        costs : list of edge weights in the same order as list(H.edges())
        selected_edges : list of edges to highlight in red (optional)
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

    # ----- Draw the edges based on costs -----
    edge_list = list(H.edges())
    for (e, cost) in zip(edge_list, costs):
        u, v = e
        x1, y1 = positions[u]
        x2, y2 = positions[v]
        plt.plot([x1, x2], [y1, y2],
                 color=cmap(normalize(cost)),
                 alpha=0.35,
                 linewidth=1.8)

    # ----- Draw selected edges in red -----
    if selected_edges is not None:
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



def draw_tree_with_weights(T, weights, title="Tree with weights"):
    """
    Draws the tree T with edge weights in a clear and readable way.

    - Edges with weight = 0 are drawn in gray.
    - Positive weights use a viridis heatmap scale.
    - Shows edge labels.
    """

    plt.figure(figsize=(10, 7))

    # Layout: spring is intuitive for arbitrary trees
    pos = nx.spring_layout(T, seed=42)

    # Extract weights in order of T.edges()
    edge_list = []
    edge_colors = []
    edge_widths = []

    cmap = sns.color_palette("viridis", as_cmap=True)

    # Collect positive weights for normalization
    pos_weights = [w for w in weights.values() if w > 0]
    if len(pos_weights) > 0:
        wmin, wmax = min(pos_weights), max(pos_weights)
        normalize = lambda w: (w - wmin) / (wmax - wmin + 1e-9)
    else:
        normalize = lambda w: 0.5

    for (u, v) in T.edges():
        key = tuple(sorted((u, v)))
        w = weights.get(key, 0.0)

        edge_list.append((u, v))

        if w == 0:
            edge_colors.append("gray")
            edge_widths.append(1.2)
        else:
            edge_colors.append(cmap(normalize(w)))
            edge_widths.append(2.5)

    # Draw nodes
    nx.draw_networkx_nodes(T, pos, node_size=600, node_color="white", edgecolors="black")

    # Draw edges with correct colors
    nx.draw_networkx_edges(T, pos, edgelist=edge_list,
                           edge_color=edge_colors, width=edge_widths, alpha=0.9)

    # Node labels
    nx.draw_networkx_labels(T, pos, font_size=10)

    # Edge labels
    edge_labels = {edge: f"{weights.get(tuple(sorted(edge)), 0):.2f}" for edge in T.edges()}
    nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels, font_color="black")

    plt.title(title)
    plt.axis("off")
    plt.show()