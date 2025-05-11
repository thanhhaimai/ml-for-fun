from dataclasses import dataclass
from typing import Self

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence

from learning.learner import BatchResult, Learner
from learning.names_classifier.names_classifier_dataset import NameSample


@dataclass
class Batch:
    samples: list[NameSample]

    @classmethod
    def from_samples(cls, batch: list[NameSample]) -> Self:
        return cls(samples=batch)


class SequentialBatchLearner(Learner[Batch]):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> BatchResult:
        outputs: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        batch_loss = torch.tensor(0.0)

        for sample in batch.samples:
            # output: [1, C]
            output = self.model(sample.input)
            outputs.append(output)
            labels.append(sample.label)
            batch_loss += self.criterion(output, sample.label)

        # outputs: [N, C]
        outputs_tensor = torch.cat(outputs, dim=0)
        labels_tensor = torch.cat(labels, dim=0)

        return BatchResult(
            outputs=outputs_tensor,
            labels=labels_tensor,
            loss=batch_loss,
            loss_scale=len(batch.samples),
        )


class ParallelBatchLearner(Learner):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> BatchResult:
        # sample.input: [S, 1, V]
        # inputs: [N, S, V]
        inputs = pad_sequence([sample.input for sample in batch.samples]).squeeze(2)
        # sample.label: [1]
        # labels: [N]
        labels = torch.cat([sample.label for sample in batch.samples])
        # shape: [N, C]
        outputs = self.model(inputs)

        batch_loss = self.criterion(outputs, labels)

        return BatchResult(
            outputs=outputs,
            labels=labels,
            loss=batch_loss,
            loss_scale=1,
        )
