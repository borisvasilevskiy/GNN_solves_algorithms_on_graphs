# GNNs and graph planarity detection

This note contains an overview of several papers related to planarity detection via GNNs.

## Graph planarity introduction

An undirected graph is called 'planar' if it can be drawn on a plane without edge intersections. There are well-established mathematical criteria for graph planarity. One of them is Kuratowski's criteria, which states that graph planarity is equivalent to the absence of a subgraph that is a subdivision of K_5 (a complete graph on 5 vertices) or K_33 (a complete bipartite graph with 3+3 vertices). A subdivision of a graph results from inserting vertices into edges zero or more times. Details can be found here: https://en.wikipedia.org/wiki/Planar_graph.

There are several planarity detection algorithms that work in linear time O(n), where n is the number of vertices. You may wonder why it's not O(E), where E is the number of edges, since that's the minimum required to read the graph. Well, there's a theorem (https://en.wikipedia.org/wiki/Planar_graph#Other_criteria) that claims the number of edges in a planar graph is <= 3n - 6. This means we won't have to read more than 3n - 6 edges of the graph to establish its planarity.

Let's note that Kuratowski's criteria is almost local. By 'local,' I mean that a constant-length neighborhood of each vertex needs to be examined for the planarity check. By 'almost,' I mean it's not completely local because it has to deal with subdivisions, i.e. smooth all vertices with power=2 from the subgraph.

## Motivation to solve planarity with GNNs

The fact that the task is algorithmically solvable makes it an easy benchmark since the dataset can be auto-generated and can contain any number of graphs with various properties.

The fact that it's almost local makes it a good candidate for GNNs to solve it with high accuracy. For instance, we can generate a dataset to contain only graphs that are solvable with the local criteria. K_5 is located in the 1-hop neighborhood (it's called a '1-disc') of any of its vertices, and K_33 is located in the 2-hop neighborhood. It means that 2 layers of GNN may already be sufficient to learn the local Kuratowski's criteria. A couple more layers won't do any harm, of course, but the point is that GNN has access to all the information needed to solve the task perfectly.

Besides that, it's interesting to learn how GNN would deal with the main task that's not local.

# Literature overview

## A Property Testing Framework for the Theoretical Expressivity of Graph Kernels
by Nils M. Kriege, Christopher Morris, Anja Rey, Christian Sohler / Proceedings of the Twenty-Seventh International Joint Conference on Artificial Intelligence / Main track. Pages 2348-2354. https://doi.org/10.24963/ijcai.2018/325

A graph kernel is a function that calculates a numeric graph similarity based on two given graphs:

k(G_1, G_2) -> real number >= 0

One connection between graph kernels and GNNs is the following: imagine we have a kernel, then we can employ GNNs to embed a graph into a d-dim space so that the graph similarity provided by the kernel is close to the scalar product of the produced embeddings. That can be done using ML methods, particularly with GNNs.

Different kernels produce different graph embeddings that can be later used to solve graph classification tasks. For instance, one can train a logistic regression that takes graph's embedding an an input and tries to predict graph's planarity. One can go with several FC layers or even with more advanced architecture. In this approach, the graph embedding is generated via fully unsupervised way, and then used as an input feature to a supervised task.

One of the paper's contribution that is of interest here, is that they consider several popular kernels and several popular graph properties (such as planarity) and establish theoretical abilities to determine these properties using the embeddings generated with the kernel.

Here're some of their results (cited from the paper):

>* The random walk kernel and the Weisfeiler–Lehman subtree
> kernel both fail to identify connectivity, planarity, bipartiteness and triangle-freeness (see Theorems 4.1, 4.2).
>* The graphlet kernel can identify triangle-freeness, but fails
>to distinguish any graph property (see Theorem 4.5).
>* We define the k-disc kernel and show that it is able to
>distinguish connectivity, planarity, and triangle-freeness
>(see Theorem 5.4).

Therefore, if one is going to embed a graph using unsupervised learning kernels, a good kernel to start with is 'k-dics'.
