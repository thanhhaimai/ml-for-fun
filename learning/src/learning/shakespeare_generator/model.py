from dataclasses import dataclass
from typing import Self

import torch
from torch import nn, optim

from learning.learner import BatchResult, Learner
from learning.shakespeare_generator.shakespeare_dataset import Sample


class ShakespeareGenerator(nn.Module):
    def __init__(
        self,
        input_size: int,
        padding_idx: int,
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding = nn.Embedding(input_size, input_size, padding_idx=padding_idx)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        indices: [B, S]
        """
        # embedding: [V, V]
        # indices: [B, S] -- advance indexing, replace the first V shape with [B, S]
        # output: [B, S, V]
        output = self.embedding(indices)
        return output

    def generate(self, indices: torch.Tensor, max_length: int) -> torch.Tensor:
        """
        indices: [B, S] -- the batched sequence of indices
        return: [B, S + max_length] -- after generated max_length tokens
        """
        V = self.input_size
        for _ in range(max_length):
            B, S = indices.shape
            # shape: [B, S, V]
            output = self(indices)
            # print(output.shape)
            if output.shape != (B, S, V):
                raise ValueError(
                    f"Invalid {output.shape=}, expected: ({B=}, {S=}, {V=})"
                )

            # shape: [B, V]
            last_output = output[:, -1, :]
            # print(last_output.shape)
            if last_output.shape != (B, V):
                raise ValueError(
                    f"Invalid {last_output.shape=}, expected: ({B=}, {V=})"
                )

            # shape: [B, V]
            probs = torch.softmax(last_output, dim=-1)
            # print(probs.shape)
            if probs.shape != (B, V):
                raise ValueError(f"Invalid {probs.shape=}, expected: ({B=}, {V=})")

            # shape: [B, 1]
            next_token_indices = torch.multinomial(probs, num_samples=1)
            # print(next_token_indices.shape)
            if next_token_indices.shape != (B, 1):
                raise ValueError(
                    f"Invalid {next_token_indices.shape=}, expected: ({B=}, 1)"
                )

            # shape: [B, S + 1]
            indices = torch.cat([indices, next_token_indices], dim=-1)
            # print(indices.shape)
            if indices.shape != (B, S + 1):
                raise ValueError(
                    f"Invalid {indices.shape=}, expected: ({B=}, {S + 1=})"
                )

        return indices


@dataclass
class Batch:
    samples: list[Sample]

    @classmethod
    def from_samples(cls, batch: list[Sample]) -> Self:
        return cls(samples=batch)


class ParallelBatchLearner(Learner[Batch]):
    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        padding_idx: int,
    ):
        super().__init__(model, optimizer, criterion)
        self.padding_idx = padding_idx
        assert self.criterion.reduction == "sum", "Reduction must be 'sum'"

    def batch_step(self, batch: Batch) -> BatchResult:
        # shape: [B, S, V]
        inputs = torch.stack([sample.input for sample in batch.samples], dim=0)
        # shape: [B, S]
        labels = torch.stack([sample.label for sample in batch.samples], dim=0)

        outputs = self.model(inputs)

        # `sum` mode
        batch_loss = self.criterion(outputs, labels)

        return BatchResult(
            outputs=outputs,
            labels=labels,
            loss=batch_loss,
            loss_scale=len(batch.samples),
        )
