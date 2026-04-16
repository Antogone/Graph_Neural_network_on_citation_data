import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayerScratch(nn.Module):
    """
    Single GCN layer implemented from scratch.
    Computes: H = activation(A_hat @ X @ W + b)

    A_hat is the normalised adjacency matrix with self-loops:
    A_hat = D^{-1/2} (A + I) D^{-1/2}
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        # Learnable weight matrix W
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Learnable bias
        self.b = nn.Parameter(torch.FloatTensor(out_features))
        # Initialise weights — Xavier uniform is standard for GNNs
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, X, A_hat):
        """
        X:     node feature matrix (N x in_features)
        A_hat: normalised adjacency matrix (N x N)
        """
        # Step 1 — linear transformation of features
        XW = X @ self.W  # (N x in_features) @ (in_features x out_features)
        # = (N x out_features)
        # Step 2 — neighbourhood aggregation
        AXW = A_hat @ XW  # (N x N) @ (N x out_features)
        # = (N x out_features)
        # Step 3 — add bias
        return AXW + self.b


class GCNScratch(nn.Module):
    """
    2-layer GCN implemented from scratch without PyG.

    Architecture:
        Input (N x F)
        → GCNLayer1 → ReLU → Dropout  (N x hidden)
        → GCNLayer2 → LogSoftmax      (N x classes)
    """

    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.layer1 = GCNLayerScratch(in_channels, hidden_channels)
        self.layer2 = GCNLayerScratch(hidden_channels, out_channels)
        self.dropout = dropout

    def precompute_A_hat(self, edge_index, num_nodes):
        """
        Compute normalised adjacency matrix from edge_index.
        This mirrors what GCNConv does internally.

        Steps:
        1. Build adjacency matrix A from edge_index
        2. Add self-loops: A_tilde = A + I
        3. Compute degree matrix D
        4. Normalise: A_hat = D^{-1/2} A_tilde D^{-1/2}
        """
        # Build adjacency matrix from edge_index
        # edge_index shape: (2, num_edges) — row 0 = source, row 1 = target
        A = torch.zeros(num_nodes, num_nodes)
        A[edge_index[0], edge_index[1]] = 1.0

        # Add self-loops
        A_tilde = A + torch.eye(num_nodes)

        # Degree matrix — row sums of A_tilde
        degrees = A_tilde.sum(dim=1)

        # D^{-1/2}
        D_inv_sqrt = torch.diag(degrees.pow(-0.5))

        # Normalised adjacency
        A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt

        return A_hat

    def forward(self, x, edge_index):
        # Precompute A_hat from edge_index
        A_hat = self.precompute_A_hat(edge_index, x.size(0))

        # Layer 1
        h = self.layer1(x, A_hat)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # Layer 2
        h = self.layer2(h, A_hat)

        return F.log_softmax(h, dim=1)

if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid
    import sys
    sys.path.append("src")
    from models.gcn import GCN

    # Load Cora
    dataset = Planetoid(root="data/", name="Cora")
    data    = dataset[0]

    # Train scratch GCN
    model_scratch = GCNScratch(
        dataset.num_features, 64, dataset.num_classes
    )
    optimizer = torch.optim.Adam(
        model_scratch.parameters(), lr=0.01, weight_decay=5e-4
    )

    print("Training GCN from scratch...")
    model_scratch.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out  = model_scratch(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            model_scratch.eval()
            with torch.no_grad():
                pred = model_scratch(
                    data.x, data.edge_index
                ).argmax(dim=1)
                val_acc = (
                    pred[data.val_mask] == data.y[data.val_mask]
                ).float().mean()
            print(f"Epoch {epoch+1} | Loss: {loss:.4f} | Val: {val_acc:.4f}")
            model_scratch.train()

    # Final test accuracy
    model_scratch.eval()
    with torch.no_grad():
        pred     = model_scratch(data.x, data.edge_index).argmax(dim=1)
        test_acc = (
            pred[data.test_mask] == data.y[data.test_mask]
        ).float().mean()

    print(f"\nGCN from scratch test accuracy: {test_acc:.4f}")
    print(f"GCNConv test accuracy:          0.8080")
    print(f"Difference:                     {abs(test_acc - 0.8080):.4f}")
    print("\nIf difference < 0.02, implementation is correct ✓")