# Graph Neural Networks tutorials

Here I provide materials I found useful to start with GNNs. Qualification expectations: MCs degree in computer science or related field; familiarity with general deep learning approaches and building blocks.

I list them in the order I happened to study them.

## Sanchez-Lengeling, et al., "A Gentle Introduction to Graph Neural Networks", Distill, 2021.

https://distill.pub/2021/gnn-intro/
DOI 10.23915/distill.00033

The article is indeed gentle. It starts with what graphs are, with a lot of examples from real life. They discuss possible tasks on graphs: a scent of a molecule; relations between two segments of an image/video.

An input of a GNN is a graph that consists of nodes, edges and a generic graph information, namely ‘master node’. Each entity has its value vector of a fixed size. It may be an embedding calculated from image processing or any other embedding calculated by a preceding NN. It may also be a 1-hot encoding. Basically, anything that you could think of to encode your input.

## Stanford CS22W by Jure Leskovec at al

http://web.stanford.edu/class/cs224w/; Available on [youtube](https://youtu.be/JAB_plj2rbA).

I highly recommend it. A profound course that doesn't dive into GNNs rightaway. First lectures explain traditional approaches (i.e., before NNs) which is very nice to have to understands GNNs.

## GNN short introduction

Here are some main terms and formulas I provide for my own reference.

There's a variety of GNN types: [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/). Let's consider only GNNs that deal with node embeddings only and don't learn embeddings for edges or the graph or take any edge features into account. I shall call them 'node-centric GNNs'.

Summarizing knowledge from [Stanford CS22W](http://web.stanford.edu/class/cs224w/) and papers listed below, I suppose I can give the following working definition for 'node-centric GNNs'.

Consider a graph G = (V, E) and a node features $h_0(v) \in \mathbb{R}^k$, $v \in V$. Let $N(v)$ be a set of all neighbour nodes of the node $v$. A GNN consists of an Encoder and Decoder parts. The Encoder calculates embeddings for each node. The Decoder uses that embeddings to solve a specific ML task.

Following [2], let's define an L-layer GNN encoder is defined by L pairs of functions ($aggregate_i$, $combine_i$), where $i = 1..L$ in the following way:

$$
  h_i(v) = combine_i ( h_{i-1}(v), aggregate_i( \{ h_{i-1}(u): u \in N(v) \} ) )
$$

The function $aggregate_i$ takes a _multiset_ of node embeddings as an input to make sure it doesn't depend on the order of neighboring nodes.

The decoder design depends on the task outcome set which is either of those:

1. each node
2. each edge
3. the whole graph

(I haven't seen a task that requires a mix of those, however it might be possible).

Typically, the task is either a classification or a regression.

In case of 'each node' outcome, the decoder is just a NN which input is a node embedding. Usually it is several-layer MLP.

For the case of 'each edge', Stanford CS22W mentiones two most common approaches that take edge's node embeddings and:

1. Concatenate the two node embeddings + MLP
2. Linear Dot-product of the two node embeddings ($h(v_1) \cdot W \cdot h(v_2)$, where $W$ is a matrix and $\cdot$ is a matrix multiplication.

In the case of 'the whole graph' there are different approaches known:

1. Introduce a virtual node that's connected with all V or a subset of V. The resulting embedding of this virtual node is considered to be an embedding of the whole graph. We then use MLP to convert an embedding into a prediction.
2. Aggregate node embeddings. The simplest way is to use SUM or MEAN or coorinate-wise MAX. A more sophisticated approach could be to split graph nodes into hierarhical clusters and aggregate node embeddings hierarhically, doing MEAN over all cluster's node and then applying an MLP layer [1]. It's claimed to be more efficient.

## Links

[1] Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec. _Hierarchical Graph Representation Learning with Differentiable Pooling_ NeurIPS 2018;  arXiv:1806.08804.

[2] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. _How powerful are graph neural networks?_ ICLR 2019, CoRR, abs/1810.00826, 2018. [arXiv:1810.00826 [cs.LG]](https://arxiv.org/abs/1810.00826)
