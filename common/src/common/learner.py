import math

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

    def train(self, train_loader: DataLoader, train_metrics: list[Metric]) -> float:
        self.model.train()
        epoch_loss = 0
        for inputs, labels in train_loader:
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            for metric in train_metrics:
                metric.update(outputs, labels, loss)

        return epoch_loss / len(train_loader)

    def eval(self, eval_loader: DataLoader, eval_metrics: list[Metric]) -> float:
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()

                for metric in eval_metrics:
                    metric.update(outputs, labels, loss)

        return epoch_loss / len(eval_loader)

    def final_eval(self, eval_loader: DataLoader, eval_metrics: list[Metric]) -> float:
        self.model.eval()
        epoch_loss = 0
        with torch.no_grad():
            for inputs, labels in eval_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                epoch_loss += loss.item()

                for metric in eval_metrics:
                    metric.update(outputs, labels, loss)

            for metric in eval_metrics:
                metric.on_epoch_complete(0)

        return epoch_loss / len(eval_loader)

    def fit(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        num_epochs: int,
        patience: int | None,
        train_metrics: list[Metric],
        eval_metrics: list[Metric],
    ) -> tuple[list[float], list[float]]:
        best_eval_loss = math.inf
        train_losses = []
        eval_losses = []
        current_patience = 0
        for epoch in range(num_epochs):
            train_loss = self.train(train_loader, train_metrics)
            eval_loss = self.eval(eval_loader, eval_metrics)

            for metric in train_metrics:
                metric.on_epoch_complete(epoch)

            for metric in eval_metrics:
                metric.on_epoch_complete(epoch)

            print(
                f"{epoch}/{num_epochs} \tTrain loss \t{train_loss:.4f} \tEval loss \t{eval_loss:.4f}"
            )

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            if patience is None:
                continue

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                current_patience = 0
            else:
                current_patience += 1

            if current_patience >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return train_losses, eval_losses

    def predict(self, input: torch.Tensor) -> int:
        _, idx = self.predict_topk(input, k=1)
        return int(idx.item())

    def predict_topk(
        self, input: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.model.eval()
        with torch.no_grad():
            output = self.model(input)
            likelihoods, indices = torch.topk(output, k=k)
            likelihoods = likelihoods.exp() / output.exp().sum()
            return likelihoods, indices
