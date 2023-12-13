from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset
import networkx as nx
import torch
from utils import pygraph_to_nx_graph
from typing import Optional, Callable


class FakeDatasetIsPlanar(FakeDataset):
    r"""A fake dataset that returns randomly generated
    :class:`~torch_geometric.data.Data` objects with binary labels
    where 0 corresponds to non-planar graphs and 1 to planar ones.

    The planarity is calculated using open-source library `"networkx"` and
    based on the Left-Right Planarity Test (see networkx documentation for
    more details).

    Args:
        num_graphs (int): The number of graphs.
        avg_num_nodes (int, optional): The average number of nodes in a graph.
            (default: :obj:`1000`)
        avg_degree (int, optional): The average degree per node.
            (default: :obj:`10`)
        edge_dim (int, optional): The number of edge features.
            (default: :obj:`0`)
        is_undirected (bool, optional): Whether the graphs to generate are
            undirected. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        **kwargs (optional): Additional attributes and their shapes
            *e.g.* :obj:`global_features=5`.
    """

    def __init__(
        self,
        num_graphs: int,
        avg_num_nodes: int = 1000,
        avg_degree: int = 10,
        edge_dim: int = 0,
        is_undirected: bool = True,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            num_graphs=num_graphs,
            avg_num_nodes=avg_num_nodes,
            avg_degree=avg_degree,
            num_channels=0,
            num_classes=0,
            edge_dim=edge_dim,
            task="graph",
            is_undirected=is_undirected,
            transform=transform,
            **kwargs,
        )

    def generate_data(self) -> Data:
        graph = super().generate_data()

        nx_temp_graph = pygraph_to_nx_graph(graph)
        is_planar, certificate = nx.check_planarity(nx_temp_graph)

        graph.y = torch.tensor([1 if is_planar else 0], dtype=torch.long)
        graph.is_planar = is_planar
        if is_planar:
            positions = nx.combinatorial_embedding_to_pos(
                certificate, fully_triangulate=False)
            positions = [p for _, p in sorted(positions.items())]
            graph.pos = torch.Tensor(positions)
        else:
            graph.pos = torch.zeros((graph.num_nodes, 2))

        return graph
