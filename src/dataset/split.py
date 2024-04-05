from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import Tuple


def split_dataset(dataset: Dataset, share=0.8, batch_size=64,
                  shuffle=True) -> Tuple[DataLoader, DataLoader]:
    if shuffle:
        dataset = dataset.shuffle()
    N = len(dataset)
    train_count = int(N*share)
    train_ds_small = dataset[:train_count]
    test_ds_small = dataset[train_count:]

    print(f'Number of training graphs: {len(train_ds_small)}')
    print(f'Number of test graphs: {len(test_ds_small)}')

    train_loader = DataLoader(
        train_ds_small, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_ds_small, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
