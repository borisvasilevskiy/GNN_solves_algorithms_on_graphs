# Literacure overview

This note concentrates on the grah planarity detection problem via GNNs. It contains:

* graph planarity introduction and motivation to solve it via GNNs
* literature overview of existing (May 2023) papers related to GNNs solving graph algorithmic tasks
* takeaways from the literature overview and high-level plans of the work I'm going to do

## Graph planarity introduction

An undirected graph is called 'planar' if it can be drawn on a plane without edge intersections. There are well-established mathematical criteria for graph planarity. One of them is Kuratowski's criteria, which states that graph planarity is equivalent to the absence of a subgraph that is a subdivision of K_5 (a complete graph on 5 vertices) or K_33 (a complete bipartite graph with 3+3 vertices). A subdivision of a graph results from inserting vertices into edges zero or more times. Details can be found here: https://en.wikipedia.org/wiki/Planar_graph.

There are several planarity detection algorithms that work in linear time O(n), where n is the number of vertices. You may wonder why it's not O(E), where E is the number of edges, since that's the minimum required to read the graph. Well, there's a theorem (https://en.wikipedia.org/wiki/Planar_graph#Other_criteria) that claims the number of edges in a planar graph is <= 3n - 6. This means we won't have to read more than 3n - 6 edges of the graph to establish its planarity.

Let's note that Kuratowski's criteria is almost local. By 'local,' I mean that a constant-length neighborhood of each vertex needs to be examined for the planarity check. By 'almost,' I mean it's not completely local because it has to deal with subdivisions, i.e. smooth all vertices with power=2 from the subgraph.

## Motivation to solve planarity with GNNs

The fact that the task is algorithmically solvable makes it an easy benchmark since the dataset can be auto-generated and can contain any number of graphs with various properties.

The fact that it's almost local makes it a good candidate for GNNs to solve it with high accuracy. For instance, we can generate a dataset to contain only graphs that are solvable with the local criteria. K_5 is located in the 1-hop neighborhood (it's called a '1-disc') of any of its vertices, and K_33 is located in the 2-hop neighborhood. It means that 2 layers of GNN may already be sufficient to learn the local Kuratowski's criteria. A couple more layers won't do any harm, of course, but the point is that GNN has access to all the information needed to solve the task perfectly.

Besides that, it's interesting to learn how GNN would deal with the main task that's not local.

# Literature overview

I haven't found any papers that benchmark different GNN approaches on the graph planarity task. However, I've found several interesting more theoretical papers. Here they are.

### A Property Testing Framework for the Theoretical Expressivity of Graph Kernels
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

## GNNs on NP-hard problems

As a very simple and not accurate explanation, an NP-hard problem is a combinatorial problem which polynomial solution is not yet found. Moreover, there's a mathematical fact that if a solution is found to one NP-hard problem then it will be possible to polynomially solve all other NP-hard problems. Examples: graph isomorphism, minimum dominating set, minimum vertex cover, maximum matching.

GNN's inference takes polynomial time relative to graph's size, GNN can't precisely solve these problems if a widely accepted assumption NP != P is true. However, it's interesting to search for more effective algorithms learned by GNNs - I think that's one of motivations.

To estimate effectiveness of an algorithm solving NP-hard problems there's a notion of approximation ratios. It is actually a success metric. I won't go into detail as I don't know much about it.

### How powerful are graph neural networks?
By Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka
CoRR, abs/1810.00826, 2018. [arXiv:1810.00826 [cs.LG]](https://arxiv.org/abs/1810.00826)

I think, it's a very good paper. I recommend reading it rather than my overview.

Authors consider a graph isomorphism test. There's a well-known Weisfeiler-Lehman (WL) graph isomorphism test [3]. There's a nice description of the test in [CS224W: Machine Learning with Graphs | 2021 | Lecture 2.3](https://youtu.be/buzsHTa4Hgs). One motivation they give is the following. Let's consider an _ideal_ GNN encoder. It then definitely maps different graphs to different embeddings and equal (isomorphic) graphs - to the same embedding. In turn, it means that it can solve graph isomorphism problem. Therefore it's natural to compare GNNs peformance with the a solid graph isomorphism detection algorithm.

It turns out that a 'node-centric' GNN can't be better than WL-test. Also, there's a certain GNN kind that as powerful as the WL-test. They propose a 'Graph Isomorphism Network' based on theoretical findings and find that it beats baselines on common graph classification benchmarks.

Let me introduce their notation first, then summarize their results.

----
#### Notation

Consider a graph G = (V, E) and a node features $h_0(v) \in \mathbb{R}^k$, $v \in V$. Let $N(v)$ be a set of all neighbour nodes of the node $v$. A GNN consists of an Encoder and Decoder parts. The Encoder calcSoulates embeddings for each node. The Decoder uses that embeddings to solve a specific ML task.

Following the paper, let's define an L-layer GNN encoder is defined by L pairs of functions ($aggregate_i$, $combine_i$), where $i = 1..L$ in the following way:

$$
  h_i(v) = combine_i ( h_{i-1}(v), aggregate_i( \{ h_{i-1}(u): u \in N(v) \} ) )
$$

The function $aggregate_i$ takes a _multiset_ of node embeddings as an input to make sure it doesn't depend on the order of neighboring nodes.

The decoder function takes node embeddings and outputs whatever is needed from the specific task. In our case, it outputs a single vector - a graph embedding.

----

Lemma 2 says that no GNN defined in the above manner can distinguish two non-isomorphic graphs that are not distinguished by the WL-test.

In Theorem 3, they state that a GNN with sufficient number of layers can be as powerful as WL-test if $combine_i$, $aggregate_i$ and $decoder$ functions are injective. An _injective_ function is any function that maps different inputs to different outputs, i.e. $f(x_1) = f(x_2) \implies x_1 = x2$.

Next, they look for specific kind of GNN that tend to converge to injective functions. The basis of that search is the Corollary 6 that claims existance of such a function $f$ that

$$
  h(c, X) = (1 + \epsilon)f(c) + \sum_{x \in X} f(x)
$$

is injective. It relies on a seemingly cute maths; I refer the interested reader to the paper.

So, they rely on MLP to perhaps learn that kind of injective function. Their final 'Graph Isomorphism Network' encoder setup is given in equation 4.1:

$$
  h_i(v) = MLP_i \left( (1 + \epsilon_i) h_{i-1}(v) + \sum\limits_{u \in N(v)} h_{i-1}(u))  \right)
$$

where $\epsilon_i$ is also a learnable scalar. It is not claimed to be the only option, by the way.

The decoder of 'Graph Isomorphism Network' considers not only the final node embeddings (the output of the layer L) but also node embeddings generated at each layer, including the input node features:

$$
  h(G) = concat_{k = 0}^L \left( \sum\limits_{v \in V} h_k(v) \right).
$$

For the graph isomorphism task it's redundant to apply MLP on $h(G)$ since it doesn't improve its ability to differentiate graph. The idea to consider embeddings from all layers comes from [4] where it's used for 'Jumping Knowledge Networks'.

They also prove that 1-layer MLP is not sufficient to effectively learn an injective function. They discuss MAX and MEAN aggregations in these terms as well and find interesting details.

#### Benchmarks

Authors benchmark two kinds of GIN:
* GIN- $\epsilon$: original GIN described above with trainble $\epsilon$ for each layer.
* GIN-0: $\epsilon$ is set to 0. Thus, it's almost GCN: the difference is in the number of MLP layers and SUM/MEAN aggregator.

Both GINs have 2-layer MLP. I didn't find mentiones about the hidden state normalization step that's used in GraphSAGE [6]. It may be important since the normalization step may hurt the injectiveness property of the message transformation function (f(x)). The answer can be found in their benchmark implementation at [github](https://github.com/weihua916/powerful-gnns).

Evaluations are performed on 9 datasets (4 bioinformatics and 5 social) against:

* theoretically less powerful GNNs:
  * with 1-layer perceptron instead of MLP
  * with MEAN aggregator or MAX aggregator
  * In particular, MEAN + 1-layer is an analogy of GCN (but not precisely it!)
  * and MAX + 1-layer is an analogy of GraphSAGE (but not precisely it!)
* several SOTAs including:
  * WL subtree kernel + C-SVM
  * Anonymous Walk Embeddings, DCNN, PATCHY-SAN, DGCNN - see the paper for references

They drop features for social datasets leaving only node degree for some of them. That's to concentrate the NN on the graph structure.

Here is the Table 1 (with 10-fold CV test results) from the paper:

![A table from this paper ([2])](images/GIN_benchmarks.png)

GINs work notably better baselines. Surprisongly, GIN-0 is strongly not worse GIN- $\epsilon$ and even outperforms it on some datasets. Here's a citation from the paper:

>Comparing GINs (GIN-0 and GIN- $\epsilon$), we observe that GIN-0 slightly but consistently outperforms GIN- $\epsilon$. Since both models fit training data equally well, the better generalization of GIN-0 may be explained by its simplicity compared to GIN- $\epsilon$.



### Approximation Ratios of Graph Neural Networks for Combinatorial Problems
by Ryoma Sato, Makoto Yamada, Hisashi Kashima
[arXiv:1905.10261 [cs.LG]](https://arxiv.org/abs/1905.10261)

Authors tell that the motivation of the paper is basically the same as of Xu et al (described above), but they consider different NP-hard problems and more than 1 of them.

The main topic of the paper is GNN performance on following NP-hard problems:
* minimum dominating set problem
* minimum vertex cover problem
* maximum matching problem

It is assumed that graphs have a bounded maximum degree (by some constant that doesn't depend on the size). As node features they mostly use one-hot encoding of node's degree.

They employ a theory of distributed local algorithms that have a lot in common with GNNs. There's actually a simple generalization of GNNs that is equivalent (in graph problem solving ability) to distributed local algorithms. The latter seems to be well-studied. With that help, they show theoretical boundaries on GNN ability to solve aforementioned NP-hard problems.

They generalize GNNs to a newly proposed Consistent Port Numbering GNNs that use port numbering (the paper contains a good definition of it).

Summarizing, GNNs are worse than simple greedy algorithms. However, in case of Minimum Vertex Cover Problem a Consistent Port Numbering GNNs happened to be able to learn a non-trivial algorithm [5]. To my mind, it's very interesting to actually train such a GNN and investigate what did it actually learn.

# Links

[1] Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec. _Hierarchical Graph Representation Learning with Differentiable Pooling_ NeurIPS 2018;  arXiv:1806.08804.

[2] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. _How powerful are graph neural networks?_ ICLR 2019, CoRR, abs/1810.00826, 2018. [arXiv:1810.00826 [cs.LG]](https://arxiv.org/abs/1810.00826)

[3] Boris Weisfeiler and A A Lehman. _A reduction of a graph to a canonical form and an algebra arising during this reduction._ Nauchno-Technicheskaya Informatsia, 2(9):12–16, 1968.

[4] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, and Stefanie Jegelka. _Representation learning on graphs with jumping knowledge networks._  In International Conference on Machine Learning (ICML), pp. 5453–5462, 2018.

[5] Matti Åstrand, Patrik Floréen, Valentin Polishchuk, Joel Rybicki, Jukka Suomela, and Jara Uitto. _A local 2-approximation algorithm for the vertex cover problem._ In Proceedings of 23rd International Symposium on Distributed Computing, DISC 2009, pages 191–205, 2009.

[6] William L Hamilton, Rex Ying, and Jure Leskovec. _Inductive representation learning on large graphs._In Advances in Neural Information Processing Systems (NIPS), pp. 1025–1035, 2017a.

[7] Muhan Zhang, Zhicheng Cui, Marion Neumann, and Yixin Chen. _An end-to-end deep learning architecture for graph classification_. In AAAI Conference on Artificial Intelligence, pp. 4438–4445, 2018.
