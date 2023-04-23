# Article overview

> https://distill.pub/2021/gnn-intro/

> Sanchez-Lengeling, et al., "A Gentle Introduction to Graph Neural Networks", Distill, 2021.

> DOI 10.23915/distill.00033

The article is indeed gentle. It starts with what graphs are, with a lot of examples from real life. They discuss possible tasks on graphs: a scent of a molecule; relations between two segments of an image/video.

An input of a GNN is a graph that consists of nodes, edges and a generic graph information, namely ‘master node’. Each entity has its value vector of a fixed size. It may be an embedding calculated from image processing or any other embedding calculated by a preceding NN. It may also be a 1-hot encoding. Basically, anything that you could think of to encode your input.

Then they go with the simplest GNN example, followed by pooling notion. This pooling operation a generalisation of the pooling operation in computer vision. I also didn’t find any equivalents of convolution operation, probably for a good reason. The GNN they build in this section consists of two parts.

The first one is N ‘GNN’ layers. GNN layer consists of node layer, edge layer and a master node layer. Embedding from each node is being processed by the same node layer, the resulting output is a new embedding of the same node. Embeddings of different nodes are not interfering with each other (yet), as well as embeddings of two or more edges,  a node and an edge, etc.

The output of the Nth layer looks like the initial input: embeddings for nodes, edges and a master node. The second part contains a pooling. It depends on the task and the actual data we have. E.g., we can have only edge embeddings but want to build node-level predictions or a graph-level predictions. It’s possible with pooling that transfers information between entities. It works pretty much as a CV pooling: gathers embeddings from neighbours, then does average / sum (their usual choice) / whatever you like, then puts that into a few dense layers, resulting with an embedding OR a desired prediction. Such an approach utilises the local graph structure (neighbours of each entity).

Next, passing messages between nodes and edges is being discussed.

——

One idea is that GNNs should be good at identifying the planarity of a graph. Indeed, it’s a local property (which is existence of K_5 and K_3_3 subgraph) and therefore doesn’t require to pass a message across all graph. Even 5 layers might be enough.
