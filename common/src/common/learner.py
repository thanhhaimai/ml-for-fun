import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from common.metrics import Metric


class Learner:
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train(self, train_loader: DataLoader, train_metrics: list[Metric]):
        self.model.train()
        for inputs, labels in train_loader:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            for metric in train_metrics:
                metric.update(outputs, labels, loss)

    def eval(self, eval_loader: DataLoader, eval_metrics: list[Metric]):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in eval_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                for metric in eval_metrics:
                    metric.update(outputs, labels, loss)

    def fit(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int,
        train_metrics: list[Metric],
        eval_metrics: list[Metric],
    ):
        for epoch in range(num_epochs):
            self.train(train_loader, train_metrics)
            self.eval(eval_loader, eval_metrics)

            for metric in train_metrics:
                metric.on_epoch_complete(epoch)

            for metric in eval_metrics:
                metric.on_epoch_complete(epoch)
