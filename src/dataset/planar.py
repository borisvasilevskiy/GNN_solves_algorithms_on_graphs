from torch_geometric.data import Data
from torch_geometric.datasets import FakeDataset
import networkx as nx
import torch
from utils import pygraph_to_nx_graph


class FakeDatasetIsPlanar(FakeDataset):

  # TBD: restrict to only graph level tasks

  def generate_data(self) -> Data:
    graph = super().generate_data()

    nx_temp_graph = pygraph_to_nx_graph(graph)
    is_planar, certificate = nx.check_planarity(nx_temp_graph)

    graph.y = torch.Tensor([1 if is_planar else 0])
    graph.is_planar = is_planar
    if is_planar:
      positions = nx.combinatorial_embedding_to_pos(certificate, fully_triangulate=False)
      positions = [p for _, p in sorted(positions.items())]
      graph.pos = torch.Tensor(positions)
    else:
      graph.pos = torch.zeros((graph.num_nodes, 2))

    return graph
