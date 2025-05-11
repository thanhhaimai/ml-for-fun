import math
import time
from abc import abstractmethod
from dataclasses import dataclass

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from learning.metrics import Metric


@dataclass
class Batch:
    # list[torch.Tensor] -- [S, N, D]
    inputs: list[torch.Tensor]
    # shape: [N]
    labels: torch.Tensor


@dataclass
class Sample:
    """
    S: sequence_length
    V: num_vocab
    C: num_classes
    """

    # shape: [S, V] -- one-hot encoded sequence
    input: torch.Tensor
    # shape: [1] -- class index
    label: torch.Tensor


class Learner:
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    @abstractmethod
    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, int]:
        pass

    @abstractmethod
    def collate_batch(self, batch: list[Sample]) -> Batch:
        pass

    def epoch_step(
        self, dataloader: DataLoader, metrics: list[Metric], train: bool
    ) -> float:
        epoch_loss = 0
        batch: Batch
        for batch in dataloader:
            outputs, batch_loss, loss_scale = self.batch_step(batch)

            if train:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            for metric in metrics:
                metric.update(outputs, batch.labels)

            epoch_loss += batch_loss.item() / loss_scale

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

            print(
                f"{epoch}/{num_epochs} -- {time.time() - start_time:.2f}s \tTrain loss \t{train_loss:.4f} \tEval loss \t{eval_loss:.4f}"
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
