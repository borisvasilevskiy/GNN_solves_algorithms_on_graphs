# GNNs and algorithmically solvable tasks

This is an educational project with the goal of understanding the capabilities of GNNs and exploring their current theoretical foundations. The idea is to engineer a GNN that can solve specific graph problems for which algorithmic solutions already exist. Here's my motivation:

1. To the best of my knowledge, GNNs are currently not quite capable of solving non-trivial algorithmic tasks on graphs (see the Literature overview). By solving these tasks, we might discover new insights that could be useful for addressing well-established challenges in the field of GNNs.

2. If we succeed in step 1, it will be interesting to observe what the GNN actually learned, as it should have acquired some algorithmic knowledge.

## Contents

* [GNN tutorials](GNN_tutorials.md)
* [Literature overview](GNN_and_algorithmically_solvable_tasks.md)
* [GNN benchmarks on graph planarity detection problem](GNN_benchmarks_on_graph_planarity_detection.md)

## Summary - GNN benchmarks on graph planarity detection problem

This activity is not finished yet. However, there're some results already. Please refer to the [report](GNN_benchmarks_on_graph_planarity_detection.md) for details.

* Synthetic 8-node dataset was solved very well by a SAGE model with 3 layers. Analysis with GNN Explaniner on the one-hot feature inputs showed that this dataset is too easy for the model so it overfits to some heuristic and doesn't really solve the planarity problem. Well, we kind of expected that :)
  * Next steps: explain the best model on Const-feature datasets.
* Syntheric 15-node dataset was solved notably worse (expectedly). With 1 constant feature input, the best model has 5 layers and it's performance is similar to the non-GNN baseline with one-hot inputs => at least, it learns to differentiate nodes.
* Increasing layers makes worse despite the fact that theoretically the problem should be solved with 15 layers (the number of nodes) with high accuracy. Most probably, that's the lack of model config / hyperparameters and a good next step.
