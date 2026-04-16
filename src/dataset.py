import torch
from torch_geometric.datasets import Planetoid
import networkx as nx
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

# ── Load Cora ─────────────────────────────────────────────────────────────────
dataset = Planetoid(root="data/", name="Cora")
data    = dataset[0]

print(f"Dataset: {dataset}")
print(f"Nodes:        {data.num_nodes}")
print(f"Edges:        {data.num_edges}")
print(f"Features:     {data.num_node_features}")
print(f"Classes:      {dataset.num_classes}")
print(f"Train nodes:  {data.train_mask.sum().item()}")
print(f"Val nodes:    {data.val_mask.sum().item()}")
print(f"Test nodes:   {data.test_mask.sum().item()}")



# ── Visualise a subgraph ──────────────────────────────────────────────────────
G = to_networkx(data, to_undirected=True)

# Sample a connected subgraph around node 0 (2-hop neighbourhood)
nodes_2hop = list(nx.ego_graph(G, 0, radius=2).nodes())[:50]
subgraph   = G.subgraph(nodes_2hop)

# Find the node whose 2-hop neighbourhood has the most class diversity
best_node    = 0
best_entropy = 0

for n in range(200):  # check first 200 nodes
    ego_nodes = list(nx.ego_graph(G, n, radius=2).nodes())[:80]
    if len(ego_nodes) < 20:
        continue
    classes   = [data.y[i].item() for i in ego_nodes]
    counts    = np.bincount(classes, minlength=7)
    probs     = counts / counts.sum()
    entropy   = -np.sum(probs * np.log(probs + 1e-9))
    if entropy > best_entropy:
        best_entropy = entropy
        best_node    = n

print(f"Best node: {best_node}, entropy: {best_entropy:.3f}")

# Try a more connected node
seed_node  = best_node
ego        = nx.ego_graph(G, seed_node, radius=2)
nodes_list = list(ego.nodes())[:80]
subgraph   = G.subgraph(nodes_list)

node_colors = [data.y[n].item() for n in subgraph.nodes()]

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(subgraph, seed=42)
nx.draw_networkx(
    subgraph, pos,
    node_color=node_colors,
    cmap=plt.cm.Set1,
    vmin=0, vmax=6,          # ← force colormap range 0-6 for 7 classes
    node_size=120,
    edge_color="gray",
    alpha=0.85,
    with_labels=False,
    width=0.5
)
plt.title("Cora subgraph — 2-hop neighbourhood\n(colour = paper topic)")
plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/cora_subgraph.png", dpi=150, bbox_inches="tight")
plt.close()
#2-hop neighbours that share no direct edges within this subgraph appear isolated — they are connected through paths outside the displayed subgraph.

print("Saved: outputs/cora_subgraph.png")