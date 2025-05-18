import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from learning.metrics import Metric


@dataclass
class BatchResult:
    outputs: torch.Tensor
    labels: torch.Tensor
    loss: torch.Tensor
    loss_scale: int


BatchT = TypeVar("BatchT")


class Learner(Generic[BatchT]):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    @abstractmethod
    def batch_step(self, batch: BatchT) -> BatchResult:
        pass

    def epoch_step(
        self, dataloader: DataLoader[BatchT], metrics: list[Metric], train: bool
    ) -> float:
        epoch_loss = 0
        for batch in dataloader:
            batch_result = self.batch_step(batch)

            if train:
                self.optimizer.zero_grad()
                batch_result.loss.backward()
                self.optimizer.step()

            for metric in metrics:
                metric.update(batch_result.outputs, batch_result.labels)

            epoch_loss += batch_result.loss.item() / batch_result.loss_scale

        return epoch_loss / len(dataloader)

    def train(self, dataloader: DataLoader, metrics: list[Metric]) -> float:
        self.model.train()
        return self.epoch_step(dataloader, metrics, train=True)

    def eval(self, dataloader: DataLoader, metrics: list[Metric]) -> float:
        self.model.eval()
        with torch.no_grad():
            return self.epoch_step(dataloader, metrics, train=False)

    def final_eval(self, dataloader: DataLoader, metrics: list[Metric]) -> float:
        self.model.eval()
        with torch.no_grad():
            epoch_loss = self.epoch_step(dataloader, metrics, train=False)
            for metric in metrics:
                metric.on_epoch_complete(0)

        return epoch_loss

    def fit(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
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
            start_time = time.time()
            train_loss = self.train(train_dataloader, train_metrics)
            eval_loss = self.eval(eval_dataloader, eval_metrics)

            for metric in train_metrics:
                metric.on_epoch_complete(epoch)

            for metric in eval_metrics:
                metric.on_epoch_complete(epoch)

            train_losses.append(train_loss)
            eval_losses.append(eval_loss)

            if patience is None:
                continue

            is_best = eval_loss < best_eval_loss
            if is_best:
                best_eval_loss = eval_loss
                current_patience = 0
            else:
                current_patience += 1

            print(
                f"{epoch}/{num_epochs} -- {time.time() - start_time:.2f}s "
                f"\tTrain loss \t{train_loss:.4f} "
                f"\tEval loss \t{eval_loss:.4f} "
                f"\t{'<<' if is_best else ''}"
            )

            if current_patience >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        return train_losses, eval_losses

    def predict(self, input: torch.Tensor) -> int:
        """
        seq_length: S
        input_size: D

        input: [S, 1, D]
        """
        _, idx = self.predict_topk(input, k=1)
        return int(idx.item())

    def predict_topk(
        self, input: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        seq_length: S
        input_size: D

        input: [S, 1, D]

        returns:
            likelihoods: [K]
            indices: [K]
        """
        self.model.eval()
        with torch.no_grad():
            # output: [1, C]
            output = self.model(input)
            # probabilities: [1, C]
            probabilities = torch.softmax(output, dim=1)

            # topk_logits: [1, K]
            # indices: [1, K]
            _topk_logits, indices = torch.topk(output, k=k, dim=1)

            # topk_probabilities: [1, K]
            topk_probabilities = torch.gather(probabilities, dim=1, index=indices)

            return topk_probabilities.squeeze(0), indices.squeeze(0)
