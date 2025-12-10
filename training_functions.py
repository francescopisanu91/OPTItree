import torch
import torch.nn as nn
import torch.nn.functional as F
import graphical_tool
import instance_generator
import additive_instance_generator
import instance_converter
import minimum_weighted_UBT_finder
import graph_generator
import numpy as np 
import random

# ---------------------------------------------------------
# Build adjacency matrix for the line graph
# ---------------------------------------------------------
def build_lg_adjacency(LG, LG_nodes, device='cpu'):
    N = len(LG_nodes)
    node_to_idx = {node: i for i, node in enumerate(LG_nodes)}
    A = torch.zeros((N, N), dtype=torch.float32, device='cpu')
    for u, v in LG.edges():
        i, j = node_to_idx[u], node_to_idx[v]
        A[i, j] = 1.0
        A[j, i] = 1.0
    # Add self-loops (standard GNN practice)
    A += torch.eye(N, device='cpu')
    # Normalize adjacency  Â = D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    A_hat = D_inv_sqrt @ A @ D_inv_sqrt
    return A_hat


# ---------------------------------------------------------
# Simple GCN layer
# ---------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.w = nn.Linear(in_dim, out_dim)
    def forward(self, A_hat, X):
        return self.w(A_hat @ X)


# ---------------------------------------------------------
# 2-layer GNN + final MLP applied to each node
# ---------------------------------------------------------
class LineGraphGNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),   # output = predicted weight
        )
    def forward(self, A_hat, X):
        h = F.relu(self.gcn1(A_hat, X))
        h = F.relu(self.gcn2(A_hat, h))
        out = self.mlp(h)        # applied independently per node
        return out.squeeze(-1)   # shape = (N,)


# ---------------------------------------------------------
# Build node features for LG
# ---------------------------------------------------------
def build_features(LG_nodes, predicted_weights_dict, known_weights_dict):
    """
    Features used (per node = per edge in G):
        [ known_flag, known_value ]
    All unknown nodes receive:
        known_flag = 0
        known_value = 0
    """
    X = []
    for node in LG_nodes:
        if node in known_weights_dict:  # known arc from D
            f = [1.0, known_weights_dict[node]]
        else:                           # predicted arc
            f = [0.0, 0.0]
        X.append(f)
    return torch.tensor(X, dtype=torch.float32)


# ---------------------------------------------------------
# One forward + backward step using Adam
# ---------------------------------------------------------
def create_dataset(n_inst=100,min_leaves=5,max_leaves=50):
    dataset=[]
    for _ in range(n_inst):
        # -------------------------
        # Generate instance
        # -------------------------
        value = random.randint(min_leaves, max_leaves)
        T, _, weights = additive_instance_generator.random_tree_with_n_leaves(value)
        T, weights = additive_instance_generator.tree_cubifier(T, weights)
        T, leaves, tree_weights = additive_instance_generator.relabel_tree_leaves_first(T, weights)
        D = additive_instance_generator.additive_matrix(T,leaves,tree_weights)
        D = instance_generator.perturb_additive_matrix(D, tree_weights)
        # Graphs
        G, S1, S2, weights = graph_generator.build_complete_graph_from_D(D)
        H, S1, S2 = graph_generator.build_star_graph_from_D(D)
        # Build line graph
        LG, edge_list, edge_to_idx = instance_converter.line_graph(G)
        LG_nodes = list(LG.nodes())   # each node = (u,v) edge of G
        N = len(LG_nodes)
        # -----------------------------------------------------------
        # (A) MASK should select ONLY nodes corresponding to edges touching S2
        # -----------------------------------------------------------
        def edge_touches_S2(edge):
            u, v = edge
            return (u in S2) or (v in S2)
        mask_bool = np.array([edge_touches_S2(edge) for edge in LG_nodes])
        # -----------------------------------------------------------
        # (B) TRUE VALUES = 1 if LG-node corresponds to an edge of the tree T
        # -----------------------------------------------------------
        tree_edges = set(tuple(sorted(e)) for e in T.edges())
        true_values = torch.tensor(
            [1.0 if tuple(sorted(edge)) in tree_edges else 0.0 for edge in LG_nodes],
            dtype=torch.float32
        )
        # -----------------------------------------------------------
        # Build X using mask_bool and a dummy feature (same structure as before)
        # -----------------------------------------------------------
        # You can choose your own features here — I just reuse something consistent
        X = torch.zeros((N, 2), dtype=torch.float32)
        for i, edge in enumerate(LG_nodes):
            if not mask_bool[i]:
                X[i] = torch.tensor([1.0, 0.0])  # visible (you can adjust features)
        # -----------------------------------------------------------
        # Build adjacency matrix
        # -----------------------------------------------------------
        A_hat = build_lg_adjacency(LG, LG_nodes, device='cpu')
        # Add to dataset
        dataset.append(((X, A_hat, torch.tensor(mask_bool, dtype=torch.bool)), (T,true_values)))
    return dataset


def training_loop(n_instances=1000,min_leaves=5,max_leaves=50,epochs=100,lr=1e-3, hidden=64):
    dataset=create_dataset(n_instances,min_leaves,max_leaves)
    model = LineGraphGNN(in_dim=2, hidden=hidden)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    optimizer.zero_grad()
    for epoch in range(epochs):
        loss_value = 0.0
        for (input, labels) in dataset:
            X, A_hat, mask = input
            T, true_values = labels
            if mask.sum() == 0:
                continue
            optimizer.zero_grad()
            logits = model(A_hat, X)  # shape [N, 1]
            loss = criterion(logits[mask], true_values[mask])
            loss.backward()
            optimizer.step()
            loss_value += loss.item()
        print("Loss:", loss_value / len(dataset))
    return model