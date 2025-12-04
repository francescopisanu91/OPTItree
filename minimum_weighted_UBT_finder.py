from instance_generator import weighted_graph_generator
from graphical_tool import draw_graph_with_solution
import pulp

def weighted_problem_solver(n, verbose=0, draw = False):
    # --- call instance generator of size n ---
    V, E, costs = weighted_graph_generator(n)
    if verbose > 0:
        print('\n INSTANCE GRAPH ',V, E, costs, '\n')
    # Vertex set pqrtition
    S1 = list(range(1, n+1))       # n vertices 
    S2 = list(range(n+1, 2*n-1))   # n-2 vertices

    # --- PuLP model ---
    model = pulp.LpProblem("minimum weigthed UBT", pulp.LpMinimize)

    # Variables x_ij >= 0 for each (i,j)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat=pulp.LpInteger)
        for (i, j) in E
    }

    # --- Objective function ---
    model += pulp.lpSum(
        costs[k] * x[E[k]] for k in range(len(E))
    )

    # --- Constraints ---
    # Star constraints: for every i in S1: sum_{j in S2}  x_ij = 1
    for i in S1:
        archi_da_i = [(u, v) for (u, v) in E if u == i and v in S2]
        model += pulp.lpSum(x[a] for a in archi_da_i) == 1,  f"vincolo_foglia_{i}"

    # Clique constraints: for every i in S2: sum_{j in S2} x_ij = 3
    for i in S2:
        archi_da_i = [(u, v) for (u, v) in E if u == i or v==i]   # TUTTI gli archi da i
        model += pulp.lpSum(x[a] for a in archi_da_i) == 3,  f"vincolo_nodo_interno_{i}"

    # Connection constraint: sum_{i,j in S2}  x_ij = n-3 internal edges
    model += pulp.lpSum( x[(i, j)] for (i, j) in E if i in S2 and j in S2) == n - 3, "vincolo_connessione"



    # --- Solve the instqnce ---
    model.solve()

    if verbose == 2:
        print("\n--- Model constrqints ---")
        for nome, vincolo in model.constraints.items():
            print(nome, " : ", vincolo)


    print("Status:", pulp.LpStatus[model.status])
    print("Optimal value:", pulp.value(model.objective), "\n")

    selected_edges = []
    print("UBT:")
    for (i, j), var in x.items():
        if var.value() > 1e-6:
            print(f"x[{i},{j}] = {var.value()}")
            selected_edges.append((i, j))
    if draw:
        draw_graph_with_solution(V,E,costs, S1, S2,selected_edges )
        
    return model

# weighted_problem_solver(30, verbose= 1, draw = False)


def ubt_solver_from_star_graph(H, S1, S2, LG_weights_filtered, verbose=1):
    """
    Solve the UBT (Unrooted Binary Tree) problem on the graph H
    using the weights coming from the filtered line graph predictions.
    
    Input:
        H : networkx.Graph (contains S1–S2 edges and S2–S2 edges)
        S1, S2 : partition of nodes of H
        LG_weights_filtered : numpy array of weights (aligned with H.edges())
    
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

    if len(E) != len(LG_weights_filtered):
        raise ValueError("Mismatch between number of edges in H and LG_weights_filtered.")

    # ------------------------------------------------------------------
    # 2. Associate weights to edges in correct order
    # ------------------------------------------------------------------
    costs = {}
    for k, e in enumerate(E):
        costs[e] = float(LG_weights_filtered[k])

    # ------------------------------------------------------------------
    # 3. Build PuLP model
    # ------------------------------------------------------------------
    model = pulp.LpProblem("UBT_from_linegraph", pulp.LpMinimize)

    # Decision variables x_e ∈ {0,1}
    x = { e : pulp.LpVariable(f"x_{e[0]}_{e[1]}", lowBound=0, upBound=1, cat=pulp.LpInteger)
          for e in E }

    # Objective
    model += pulp.lpSum(costs[e] * x[e] for e in E)

    # ------------------------------------------------------------------
    # 4. Constraints
    # ------------------------------------------------------------------

    # (A) Star constraints for leaves S1: exactly 1 chosen edge to S2
    for i in S1:
        edges_from_i = [(u, v) for (u, v) in E if u == i and v in S2]
        model += pulp.lpSum(x[e] for e in edges_from_i) == 1, f"leaf_{i}"

    # (B) Internal S2 nodes: degree 3
    for i in S2:
        edges_touching_i = [(u, v) for (u, v) in E if u == i or v == i]
        model += pulp.lpSum(x[e] for e in edges_touching_i) == 3, f"internal_{i}"

    # (C) Total #internal edges (S2–S2) = |S1| - 3
    n = len(S1)
    model += pulp.lpSum(x[e] for e in E if e[0] in S2 and e[1] in S2) == n - 3, "internal_edges"

    # ------------------------------------------------------------------
    # 5. Solve
    # ------------------------------------------------------------------
    model.solve()

    if verbose >= 1:
        print("Status:", pulp.LpStatus[model.status])
        print("Optimal cost:", pulp.value(model.objective))
        print("")

    selected_edges = []
    for e in E:
        if x[e].value() > 0.5:
            selected_edges.append(e)
            if verbose >= 2:
                print(f"Selected edge {e} with weight {costs[e]}")

    return model, selected_edges