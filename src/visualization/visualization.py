from torch_geometric.data import Data
from matplotlib import pyplot as plt
from utils import pygraph_to_nx_graph
import networkx as nx


def visualize(data: Data):
  nx_temp_graph = pygraph_to_nx_graph(data)

  if data.is_planar:
      pos = {idx: p.tolist() for idx, p in enumerate(data.pos)}
  else:
    pos = None
  
  nx.draw_networkx(nx_temp_graph, pos)
  plt.title(f"is_planar={bool(data.y[0])}")
