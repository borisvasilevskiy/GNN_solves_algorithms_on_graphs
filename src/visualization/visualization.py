from matplotlib import pyplot as plt
from torch_geometric.data import Data
from utils import pygraph_to_nx_graph
from typing import Optional, Dict, Tuple
import networkx as nx


EdgeWeights = Dict[Tuple[int, int], float]


def visualize(data: Data, edge_weight: Optional[EdgeWeights] = None,
              undirected: bool = True, color_alpha=0.5):
    nx_temp_graph = pygraph_to_nx_graph(data)
    if undirected:
        nx_temp_graph = nx_temp_graph.to_undirected()

    if data.is_planar:
        pos = {idx: p.tolist() for idx, p in enumerate(data.pos)}
    else:
        pos = nx.spring_layout(nx_temp_graph)

    if edge_weight is None:
        edge_color = 'black'
        widths = None
    else:
        edge_color = [edge_weight[(u, v)] for u, v in nx_temp_graph.edges()]
        widths = [abs(x) * color_alpha for x in edge_color]

    nx.draw_networkx(nx_temp_graph, pos,
                     width=widths,
                     edge_color=edge_color,
                     )

    if edge_weight is not None:
        edge_labels = {k: ('%.2f' % v) for k, v in edge_weight.items()}
        nx.draw_networkx_edge_labels(nx_temp_graph, pos,
                                     edge_labels=edge_labels,
                                     font_color='black')

    plt.title(f"is_planar={bool(data.y[0])}")
