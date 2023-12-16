import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, global_mean_pool


class ConvGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes,
                 apply_dropout=True, manual_seed=12345):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        self.apply_dropout = apply_dropout
        self.conv1 = GraphConv(num_node_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
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
