# GNN benchmarks on graph planarity detection problem

Contents:
* Synthetic dataset with 8 nodes on average
* Synthetic dataset with 15 nodes on average
* Analysed the best model predictions with GNNExplainer

## Summary
* Synthetic 8-node dataset was solved very well by a SAGE model with 3 layers. Analysis with GNN Explaniner on the one-hot feature inputs showed that this dataset is too easy for the model so it overfits to some heuristic and doesn't really solve the planarity problem. Well, we kind of expected that :)
  * Next steps: explain the best model on Const-feature datasets.
* Syntheric 15-node dataset was solved notably worse (expectedly). With 1 constant feature input, the best model has 5 layers and it's performance is similar to the non-GNN baseline with one-hot inputs => at least, it learns to differentiate nodes.
* Increasing layers makes worse despite the fact that theoretically the problem should be solved with 15 layers (the number of nodes) with high accuracy. Most probably, that's the lack of model config / hyperparameters and a good next step.

## 8 nodes

Notebook: [src/Is_Planar.ipynb](src/Is_Planar.ipynb)

**Summary**:
* The performance of Conv GNN with const feature-vectors is very similar to Non-GNN baseline with one-hot inputs (no graph structure).
* With one-hot inputs, SAGE and GCN solves the dataset very well (96% F1 and 98% Acc.)
* Increasing layers makes worse despite the fact that theoretically the problem should be solved with ~8 layers. Most probably, that's the lack of model config / hyperparameters.

This synthetic dataset is based on `torch_geometric.datasets.FakeDataset` in following settings:
* node features: one hot degree; dataset: avg_degree=4, avg_num_nodes=8
  * GNNs: GCN, SAGE, Conv (see `src/models`)
  * NNs: input - sum of node features (no edges), 4 hidden layers.
* node features: constant; identical graph dataset;
  * GNNs: GCN, SAGE, Conv
 
The number of graphs is each dataset is 1mln. Train/Test split is 80/20. No dev dataset because no hyperparameters are tuned.

Result table legend:
* F1: the value of the test F1 score after 100 epoch
* F1 10: mean F1 score over the last 10 epoches (91-100)
* Accuracy: the value of Accuracy score after 100 epoch
* One hot 8: num nodes = 8, avg degree = 4, size = 1mln, one hot degree features with 11 elements; 216652 positives.
* Const 8:   identical graph dataset to 'One Hot 8', one constant node feature = 1.0; 216652 positives.
* Baseline: A simple non-GNN model with 4 hidden dense layers and 100 hidden channels. Input: sum of node features, no edges given.
* SAGE-5 and SAGE-10: SAGE arch with 5 and 10 convolutional layers correspondingly.
* SAGE-3-20: SAGE arch with 3 convolutional layers and 20 hidden channels.

| Dataset | Model  | Test F1 | Test F1 10 | Test Accuracy | Train F1 | Train Accuracy |
|---|---|---|---|---|---|---|
| One hot 8 | Baseline | 0.862  | 0.853 | 0.943 | 0.863 | 0.943 |
| One hot 8 | GCN      | 0.929  | 0.931 | 0.969 | 0.931 | 0.970 |
| One hot 8 | SAGE     | **0.964** | 0.966 | **0.985** | 0.964 | 0.985 |
| One hot 8 | Conv     | 0.957 | 0.956 | 0.982 | 0.958 | 0.982 |
| One hot 8 | SAGE-5   | 0.869 | 0.887 | 0.947 | 0.870 | 0.947 |
| One hot 8 | SAGE-3-20| 0.880 | 0.890 | 0.943 | 0.881 | 0.943 |
| One hot 8 | SAGE-4   | 0.860 | 0.883 | 0.932 | 0.861 | 0.932 |

| Dataset | Model  | Test F1 | Test F1 10 | Test Accuracy | Train F1 | Train Accuracy |
|---|---|---|---|---|---|---|
| Const 8 | GCN  | 0     | 0     | 0.783 | 0 | 0.783 |
| Const 8 | SAGE | 0     | 0     | 0.783 | 0 | 0.783 |
| Const 8 | Conv | 0.877 | 0.869 | 0.944 | 0.878 | 0.9450

## 15 nodes

Notebook: [src/Is_Planar_15_nodes.ipynb](src/Is_Planar_15_nodes.ipynb)

**Summary**:
* One hot 15: couldn't jump higher than the baseline on OneHot-15. Adding more layers made worse - probably, can be fixed by better model config / hyperparam tuning.
* Const 15: achieved the same performance with 5-layer SAGE-noDO as the One-hot dataset baseline. Models with 3 layers perform worse. 10 layers also makes worse.

We'll consider the following settings:
* node features: one hot degree; dataset: avg_degree=2, avg_num_nodes=15
  * GNNs: GCN, SAGE, Conv (see `src/models`), all with 3 convolutional layers.
  * SAGE with sum aggregation
  * SAGE with 5 and 10 convolutional layers 
  * NNs: input - sum of node features (no edges), 4 hidden layers.
* node features: constant; dataset: avg_degree=2, avg_num_nodes=15
  * GNNs: GCN, SAGE, Conv, SAGE with sum aggr
  * SAGE with 5 and 10 convolutional layers

Unlike the previous notebook, the number of graphs is each dataset is **100k**. Train/Test split is 80/20. No dev dataset because no hyperparameters are tuned.

Legend:
* F1: the value of the test F1 score after 100 epoch
* F1 10: mean F1 score over the last 10 epoches (91-100)
* Accuracy: the value of Accuracy score after 100 epoch
* One hot 15: num nodes = 15, avg degree = 2, size = 100k, one hot degree features with 16 elements; 31665 positives, 100k samples.
* Const 15: identical graph dataset to 'One Hot 15', one constant node feature = 1.0; 31665 positives, 100k samples.
* Baseline: A simple non-GNN model with 4 hidden dense layers and 100 hidden channels. Input: sum of node features, no edges given.
* SAGE sum: SAGE arch with sum aggregation
* SAGE-5 and SAGE-10: SAGE arch with 5 and 10 convolutional layers correspondingly.
* Conv noDO: no dropout applied.


| Dataset | Model  | Test F1 | Test F1 10 | Test Accuracy | Train F1 | Train Accuracy |
|---|---|---|---|---|---|---|
| One hot 15 | Baseline  | 0.692  | 0.694 | 0.821 | 0.693 | 0.823 |
| One hot 15 | GCN       | 0.697  | 0.683 | 0.809 | 0.699 | 0.813 |
| One hot 15 | SAGE      | **0.704**  | 0.694 | **0.824** | 0.706 | 0.827 |
| One hot 15 | Conv      | **0.704**  | 0.688 | 0.822 | 0.709 | 0.827 |
| One hot 15 | SAGE sum  | 0.593  | 0.562 | 0.788 | 0.595 | 0.791 |
| One hot 15 | SAGE-5    | 0.638  | 0.621 | 0.790 | 0.644 | 0.797 |
| One hot 15 | SAGE-10   | 0.533  | 0.607 | 0.471 | 0.530 | 0.470 |


| Dataset | Model  | Test F1 | Test F1 10 | Test Accuracy | Train F1 | Train Accuracy |
|---|---|---|---|---|---|---|
| Const  15 | GCN       | 0      | 0     | 0.684 | 0     | 0.68  |
| Const  15 | SAGE      | 0      | 0     | 0.684 | 0     | 0.68  |
| Const  15 | Conv      | 0.671  | 0.637 | 0.780 | 0.674 | 0.785 |
| Const  15 | Conv-noDO | 0.637  | 0.638 | 0.798 | 0.637 | 0.800 |
| Const  15 | SAGE sum-noDO| 0.637  | 0.638 | 0.798 | 0.637 | 0.800 |
| Const  15 | SAGE-5       | 0.386 | 0.511 | 0.745 | 0.387 | 0.748 |
| Const  15 | SAGE-5-noDO  | **0.715** | **0.709** | **0.820** | 0.719 | 0.824 |
| Const  15 | SAGE-10-noDO | 0.683 | 0.670 | 0.811 | 0.683 | 0.814 |

## GNN Explainer on SAGE-3 model, One Hot 8 dataset

Notebook: [src/Explain_GNN.ipynb](src/Explain_GNN.ipynb)

Explained predictions of SAGE-3 model on OneHot8 dataset (the one with 96% F1 and 98% Acc) using GNN Explainer.
* Generated ~20 explanation of the test dataset
* None of them highlight any subgraphs similar to K_{33} or K_5. Found subgraphs are all planar. Therefore, there's no evidence that the GNN has learned to seek for K_{33} or K_5 subgraphs.
* Conclusion: the dataset is too easy for GNN so it overfits on some heuristics and doesn't really solve the planarity problem.
