import torch
from torch_geometric.nn import SAGEConv, global_mean_pool, GraphSAGE
import torch.nn.functional as F


class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes,
                 apply_dropout=True, manual_seed=12345):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        self.apply_dropout = apply_dropout
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.apply_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


class CustomSAGE(torch.nn.Module):
    """Customizable version of the arch: one can specify the number pf
    convolutional layers and additional kwargs to be passed into the SageConv.
    """

    def __init__(self, hidden_layers, hidden_channels, num_node_features,
                 num_classes, apply_dropout=True, manual_seed=12345, **kwargs):
        super().__init__()
        if manual_seed is not None:
            torch.manual_seed(manual_seed)
        self.sage = GraphSAGE(
            in_channels=num_node_features,
            hidden_channels=hidden_channels,
            num_layers=hidden_layers,
            out_channels=hidden_channels,
            dropout=0.5 if apply_dropout else 0.0,
            act="relu",
            **kwargs,
        )
        self.apply_dropout = apply_dropout
        self.lin = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch, edge_weight=None):
        # 1. Obtain node embeddings
        x = self.sage(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        if self.apply_dropout:
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
