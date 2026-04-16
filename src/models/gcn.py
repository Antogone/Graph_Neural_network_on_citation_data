import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        # Two message-passing layers
        self.conv1   = GCNConv(in_channels, hidden_channels)
        self.conv2   = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Layer 1 — aggregate 1-hop neighbourhood
        x = self.conv1(x, edge_index)   # message passing
        x = F.relu(x)                    # non-linearity
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Layer 2 — aggregate 2-hop neighbourhood
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)