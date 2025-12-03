import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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