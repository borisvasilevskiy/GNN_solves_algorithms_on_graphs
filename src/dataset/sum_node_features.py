from torch_geometric.data import Dataset
import torch


def sum_node_features(data):
    graphs_in_batch = len(data.y)
    x_sum_over_nodes = [((data.batch == graph_id).reshape(-1, 1) * data.x)
                        .sum(axis=0) for graph_id in range(graphs_in_batch)]
    # print(len(x_sum_over_nodes), x_sum_over_nodes[0].shape)
    x = torch.stack(x_sum_over_nodes, dim=0)
    assert x.shape == (graphs_in_batch, data[0].num_node_features)
    return x


def _test_sum_node_features(train_ds: Dataset):
    from torch_geometric.loader import DataLoader
    from visualization import visualize
    from matplotlib import pyplot as plt

    train_loader_no_shuffle = DataLoader(train_ds, batch_size=5, shuffle=False)

    # Iterate in batches over the training dataset.
    for data in train_loader_no_shuffle:
        x = sum_node_features(data)
        print(x.shape)
        for graph_id in range(5):
            print(x[graph_id])
            orig_x = train_ds[graph_id].x.sum(axis=0)
            assert torch.all(orig_x == x[graph_id])
            visualize(train_ds[graph_id])
            plt.show()

        break
