from torch_geometric.data import Data
import networkx as nx


def pygraph_to_nx_graph(graph: Data):
    edge_tuples = graph.edge_index.transpose(0, 1).tolist()
    nx_graph = nx.Graph(edge_tuples)
    return nx_graph
