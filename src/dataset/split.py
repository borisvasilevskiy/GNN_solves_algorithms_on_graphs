from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from typing import Tuple


def split_dataset(dataset: Dataset, share=0.8) -> Tuple[DataLoader,
                                                        DataLoader]:
    dataset_shuffled = dataset.shuffle()
    N = len(dataset_shuffled)
    train_count = int(N*share)
    train_ds_small = dataset_shuffled[:train_count]
    test_ds_small = dataset_shuffled[train_count:]

    print(f'Number of training graphs: {len(train_ds_small)}')
    print(f'Number of test graphs: {len(test_ds_small)}')

    train_loader = DataLoader(train_ds_small, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds_small, batch_size=64, shuffle=False)

    return train_loader, test_loader
