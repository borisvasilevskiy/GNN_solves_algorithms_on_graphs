from .train_eval import TrainEval
from dataset import sum_node_features


class TrainEvalGraphless(TrainEval):
    def run_model(self, data, model):
        x = sum_node_features(data)
        return model(x)
