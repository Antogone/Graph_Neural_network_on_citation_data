import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import os, sys
from models.gcn_scratch import GCNScratch

sys.path.append("src")
from models.mlp_baseline import MLP
from models.gcn import GCN
from models.sage import GraphSAGE

os.makedirs("outputs", exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
dataset = Planetoid(root="data/", name="Cora")
data    = dataset[0]

# ── Training function ─────────────────────────────────────────────────────────
def train_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer    = torch.optim.Adam(model.parameters(),
                                     lr=lr, weight_decay=weight_decay)
    train_losses = []
    val_accs     = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out  = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred    = model(data.x, data.edge_index).argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).float().mean()

        train_losses.append(loss.item())
        val_accs.append(val_acc.item())

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss:.4f} | Val: {val_acc:.4f}")

    return train_losses, val_accs

# ── Evaluate function ─────────────────────────────────────────────────────────
def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out  = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        acc  = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
    return acc.item(), out.detach()

# ── Train all three models ────────────────────────────────────────────────────
# model_configs = {
#     "MLP":       MLP(dataset.num_features, 64, dataset.num_classes),
#     "GCN":       GCN(dataset.num_features, 64, dataset.num_classes),
#     "GraphSAGE": GraphSAGE(dataset.num_features, 64, dataset.num_classes),
# }

model_configs = {
    "MLP":            MLP(dataset.num_features, 64, dataset.num_classes),
    "GCN (PyG)":      GCN(dataset.num_features, 64, dataset.num_classes),
    "GCN (scratch)":  GCNScratch(dataset.num_features, 64, dataset.num_classes),
    "GraphSAGE":      GraphSAGE(dataset.num_features, 64, dataset.num_classes),
}

results    = {}
embeddings = {}
all_losses = {}
all_vaccs  = {}

for name, model in model_configs.items():
    print(f"\n── Training {name} ───────────────────────")
    losses, vaccs = train_model(model, data)
    test_acc, out = evaluate(model, data)

    results[name]    = test_acc
    embeddings[name] = out
    all_losses[name] = losses
    all_vaccs[name]  = vaccs

    print(f"  → Test accuracy: {test_acc:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
torch.save(embeddings, "outputs/embeddings.pt")
torch.save(data,       "outputs/data.pt")

# ── Results summary ───────────────────────────────────────────────────────────
print("\n── Final Results ────────────────────────────")
for name, acc in results.items():
    print(f"  {name:12s}: {acc:.4f}")

# ── Training curves ───────────────────────────────────────────────────────────
colors = {
    "MLP":           "#7C3AED",
    "GCN (PyG)":     "#0D9488",
    "GCN (scratch)": "#DC2626",
    "GraphSAGE":     "#D97706"
}

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for name in model_configs:
    axes[0].plot(all_losses[name], label=name, color=colors[name])
    axes[1].plot(all_vaccs[name],  label=name, color=colors[name])

axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].set_title("Validation accuracy")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig("outputs/training_curves.png", dpi=150)
plt.close()
print("\nSaved: outputs/training_curves.png")