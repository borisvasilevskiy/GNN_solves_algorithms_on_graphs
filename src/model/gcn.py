import torch
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F


class GCN(torch.nn.Module):
    """The model code is taken from PyG docs, "Graph Classification" example
    https://pytorch-geometric.readthedocs.io/en/latest/get_started/colabs.html
    """

    def __init__(self, hidden_channels, num_node_features, num_classes,
                 apply_dropout=True, manual_seed=12345):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        self.apply_dropout = apply_dropout
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.apply_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
