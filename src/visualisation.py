import torch
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
import os

os.makedirs("outputs", exist_ok=True)

# ── Load saved embeddings ─────────────────────────────────────────────────────
embeddings = torch.load("outputs/embeddings.pt")
data = torch.load("outputs/data.pt", weights_only=False)
labels     = data.y.numpy()

class_names = [
    "Case Based",
    "Genetic Algorithms",
    "Neural Networks",
    "Probabilistic Methods",
    "Reinforcement Learning",
    "Rule Learning",
    "Theory"
]

colors = ["#7C3AED", "#0D9488", "#D97706",
          "#DC2626", "#2563EB", "#059669", "#9333EA"]

# ── Plot UMAP for each model ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, emb) in zip(axes, embeddings.items()):
    emb_np = emb.numpy()

    reducer    = UMAP(n_components=2, random_state=42)
    emb_2d     = reducer.fit_transform(emb_np)

    for cls_idx in range(7):
        mask = labels == cls_idx
        ax.scatter(
            emb_2d[mask, 0],
            emb_2d[mask, 1],
            c=colors[cls_idx],
            label=class_names[cls_idx],
            s=8,
            alpha=0.7
        )

    ax.set_title(f"{name}\nTest acc: {emb_np.shape}")
    ax.set_xticks([])
    ax.set_yticks([])

# Add accuracy to titles
accs = {"MLP": 0.579, "GCN": 0.808, "GraphSAGE": 0.817}
for ax, name in zip(axes, accs):
    ax.set_title(f"{name} — test acc: {accs[name]:.1%}")

axes[0].legend(loc="lower left", fontsize=7, markerscale=2)
plt.suptitle("UMAP projection of learned node embeddings", y=1.02)
plt.tight_layout()
plt.savefig("outputs/umap_embeddings.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: outputs/umap_embeddings.png")