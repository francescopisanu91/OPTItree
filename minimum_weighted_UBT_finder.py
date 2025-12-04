from graphical_tool import draw_solution
from pulp import lpSum, LpVariable, LpStatus, value, LpProblem, LpMinimize, LpInteger

def ubt_solver_from_star_graph(H, S1, S2, predicted_weights, verbose=1):
    """
    Solve the UBT (Unrooted Binary Tree) problem on the graph H
    using the weights coming from the filtered line graph predictions.
    
    Input:
        H : networkx.Graph (contains S1–S2 edges and S2–S2 edges)
        S1, S2 : partition of nodes of H
        predicted_weights : numpy array of weights (aligned with H.edges())
    
    Output:
        model : solved PuLP model
        selected_edges : list of edges selected in the optimal UBT
    """

    # ------------------------------------------------------------------
    # 1. Extract edges in deterministic lexicographic order
    # ------------------------------------------------------------------
    E = list(H.edges())
    E = [tuple(sorted(e)) for e in E]
    E.sort()

    if len(E) != len(predicted_weights):
        raise ValueError("Mismatch between number of edges in H and predicted_weights.")

    # ------------------------------------------------------------------
    # 2. Associate weights to edges in correct order
    # ------------------------------------------------------------------
    costs = {}
    for k, e in enumerate(E):
        costs[e] = float(predicted_weights[k])

    # ------------------------------------------------------------------
    # 3. Build PuLP model
    # ------------------------------------------------------------------
    model = LpProblem("UBT_from_linegraph", LpMinimize)

    # Decision variables x_e ∈ {0,1}
    x = { e : LpVariable(f"x_{e[0]}_{e[1]}", lowBound=0, upBound=1, cat=LpInteger)
          for e in E }

    # Objective
    model += lpSum(costs[e] * x[e] for e in E)

    # ------------------------------------------------------------------
    # 4. Constraints
    # ------------------------------------------------------------------

    # (A) Star constraints for leaves S1: exactly 1 chosen edge to S2
    for i in S1:
        edges_from_i = [(u, v) for (u, v) in E if u == i and v in S2]
        model += lpSum(x[e] for e in edges_from_i) == 1, f"leaf_{i}"

    # (B) Internal S2 nodes: degree 3
    for i in S2:
        edges_touching_i = [(u, v) for (u, v) in E if u == i or v == i]
        model += lpSum(x[e] for e in edges_touching_i) == 3, f"internal_{i}"

    # (C) Total #internal edges (S2–S2) = |S1| - 3
    n = len(S1)
    model += lpSum(x[e] for e in E if e[0] in S2 and e[1] in S2) == n - 3, "internal_edges"

    # ------------------------------------------------------------------
    # 5. Solve
    # ------------------------------------------------------------------
    model.solve()

    if verbose >= 1:
        print("Status:", LpStatus[model.status])
        print("Optimal cost:", value(model.objective))
        print("")

    selected_edges = []
    for e in E:
        if x[e].value() > 0.5:
            selected_edges.append(e)
            if verbose >= 2:
                print(f"Selected edge {e} with weight {costs[e]}")

    draw_solution( H, S1, S2,predicted_weights, selected_edges )
    return model, selected_edges