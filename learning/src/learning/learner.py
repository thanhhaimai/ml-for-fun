import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
import torchinfo
from torch import nn, optim
from torch.utils.data import DataLoader

from learning.metrics import Metric


@dataclass
class Config:
    batch_size: int
    sequence_length: int
    embedding_size: int
    num_heads: int
    epochs: int
    dropout: float
    learning_rate: float
    patience: int | None
    min_delta: float | None
    device: torch.device


@dataclass
class BatchResult:
    outputs: torch.Tensor
    labels: torch.Tensor
    loss: torch.Tensor
    sample_count: int


BatchT = TypeVar("BatchT")


class Learner(Generic[BatchT]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    @abstractmethod
    def batch_step(self, batch: BatchT) -> BatchResult:
        pass

    def epoch_step(
        self, dataloader: DataLoader[BatchT], metrics: list[Metric], train: bool
    ) -> float:
        epoch_loss = 0
        epoch_samples = 0
        # with torch.autocast(device_type=self.device.type, enabled=True):
        for batch in dataloader:
            batch_result = self.batch_step(batch)

            if train:
                # with torch.autocast(device_type=self.device.type, enabled=False):
                self.optimizer.zero_grad()
                (batch_result.loss / batch_result.sample_count).backward()
                self.optimizer.step()

            for metric in metrics:
                metric.update(batch_result.outputs, batch_result.labels)

            epoch_loss += batch_result.loss.item()
            epoch_samples += batch_result.sample_count

        return epoch_loss / epoch_samples

    def train(
        self,
        dataloader: DataLoader,
        metrics: list[Metric] = [],
    ) -> float:
        self.model.train()
        return self.epoch_step(dataloader, metrics, train=True)

    def eval(
        self,
        dataloader: DataLoader,
        metrics: list[Metric] = [],
    ) -> float:
        self.model.eval()
        with torch.no_grad():
            return self.epoch_step(dataloader, metrics, train=False)

    def final_eval(
        self,
        dataloader: DataLoader,
        metrics: list[Metric] = [],
    ) -> float:
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
        min_delta: float | None,
        train_metrics: list[Metric] = [],
        eval_metrics: list[Metric] = [],
    ) -> tuple[list[float], list[float]]:
        best_eval_loss = math.inf
        train_losses = []
        eval_losses = []
        current_patience = 0
        effective_min_delta = min_delta if min_delta is not None else 0.0

        initial_eval_loss = self.eval(eval_dataloader, eval_metrics)
        print(f"Initial eval loss: {initial_eval_loss:.4f}")

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

            is_best = eval_loss < best_eval_loss - effective_min_delta
            print(
                f"{epoch}/{num_epochs} -- {time.time() - start_time:.2f}s "
                f"\tTrain loss \t{train_loss:.4f} "
                f"\tEval loss \t{eval_loss:.4f} "
                f"\t{'<<' if is_best else ''}"
            )

            if patience is None:
                continue

            if is_best:
                best_eval_loss = eval_loss
                current_patience = 0
            else:
                current_patience += 1

            if current_patience >= patience:
                print(f"Early stopping at epoch {epoch} {best_eval_loss=:.4f}")
                break

        return train_losses, eval_losses

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"\nmodel={self.model}"
            f"\n{torchinfo.summary(self.model)}"
            f"\noptimizer={self.optimizer}"
            f"\ncriterion={self.criterion}"
        )
