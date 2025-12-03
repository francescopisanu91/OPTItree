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




weighted_problem_solver(30, verbose= 1, draw = False)