from matplotlib import pyplot as plt
import logging
import numpy as np
import torch


class TrainEval(object):
    def __init__(self, model_instance, device=None):
        self.optimizer = torch.optim.Adam(model_instance.parameters(), lr=0.01)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.model_instance = model_instance
        self.device = device or torch.device('cpu')

    def run_model(self, data, model):
        return model(data.x, data.edge_index, data.batch)

    def test(self, model, loader):
        model.eval()
        # print('[before test] lin sum^2:', torch.pow(model.lin.weight, 2).sum())

        tp = 0
        fn = 0
        fp = 0
        first_batch = True
        # Iterate in batches over the training/test dataset.
        for data in loader:
            data = data.to(self.device)
            out = self.run_model(data, model)
            # print('out:', out)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            if first_batch:
                logging.debug(f'first batch out[:5]: {out[:5]}')
                first_batch = False
            tp += int(((pred == 1)*(data.y == 1)).sum())
            fn += int(((pred == 0)*(data.y == 1)).sum())
            fp += int(((pred == 1)*(data.y == 0)).sum())

        total = len(loader.dataset)
        accuracy = (total - fp - fn) / total
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2*recall*precision / (recall + precision + 1e-9)
        # print(f'total={total}, tp={tp}, fp={fp}, fn={fn}')
        return accuracy, precision, recall, total, f1

    SUPPORTED_METRIC_NAMES = ['accuracy', 'precision', 'recall', 'size', 'f1']

    def fmt(self, metric_tuple):
        output = []
        for metric_idx, metric_name in enumerate(self.SUPPORTED_METRIC_NAMES):
            metric_fmt = f"{metric_name}={metric_tuple[metric_idx]:.4f}"
            output.append(metric_fmt)

        return ", ".join(output)

    def train(self, model, criterion, optimizer, train_loader):
        model.train()

        # logging.debug(f'[before training] lin sum^2: '
        #               f'{torch.pow(model.lin.weight, 2).sum()}')

        for data in train_loader:  # Iterate in batches over the training ds
            data = data.to(self.device)
            out = self.run_model(data, model)
            # print('out:', out)
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        # logging.debug(f'[after training] lin sum^2: '
        #               f'{torch.pow(model.lin.weight, 2).sum()}')

    def draw_metrics(self, train_metrics_per_epoch, test_metrics_per_epoch):
        plt.figure(figsize=(10, 10))
        for metric_idx, metric_name in enumerate(self.SUPPORTED_METRIC_NAMES):
            plt.subplot(3, 2, metric_idx + 1)
            plt.plot([t[metric_idx] for t in train_metrics_per_epoch],
                     label='train')
            plt.plot([t[metric_idx] for t in test_metrics_per_epoch],
                     label='test')
            plt.legend()
            plt.title(metric_name)
        plt.show()

    def print_metric_summary(self, train_metrics_per_epoch,
                             test_metrics_per_epoch):
        test_f1 = test_metrics_per_epoch[-1][self.SUPPORTED_METRIC_NAMES.index(
            'f1')]
        test_f1_last10 = np.mean([item[self.SUPPORTED_METRIC_NAMES.index('f1')]
                                  for item in test_metrics_per_epoch[-10:]])
        test_accuracy = test_metrics_per_epoch[-1][
            self.SUPPORTED_METRIC_NAMES.index('accuracy')]
        train_f1 = train_metrics_per_epoch[-1][
            self.SUPPORTED_METRIC_NAMES.index('f1')]
        train_accuracy = train_metrics_per_epoch[-1][
            self.SUPPORTED_METRIC_NAMES.index('accuracy')]

        print(("| Test F1 | Test F1 10 | Test Accuracy | Train F1 "
               "| Train Accuracy | "))
        print("|---|---|---|---|---|")
        print((f"| {test_f1} | {test_f1_last10} | {test_accuracy} | {train_f1}"
               f" | {train_accuracy} |"))

    def main(self, epoch, train_loader, test_loader):
        train_metrics_per_epoch = []
        test_metrics_per_epoch = []

        for epoch_id in range(1, epoch+1):
            self.train(self.model_instance, self.criterion,
                       self.optimizer, train_loader)
            train_metrics = self.test(self.model_instance, train_loader)
            test_metrics = self.test(self.model_instance, test_loader)
            print((f'Epoch: {epoch_id: 03d}, Train: {self.fmt(train_metrics)},'
                   f' Test: {self.fmt(test_metrics)}'))
            train_metrics_per_epoch.append(train_metrics)
            test_metrics_per_epoch.append(test_metrics)

        self.draw_metrics(train_metrics_per_epoch, test_metrics_per_epoch)

        self.print_metric_summary(
            train_metrics_per_epoch, test_metrics_per_epoch)
