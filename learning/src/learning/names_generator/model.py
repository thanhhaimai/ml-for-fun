from dataclasses import dataclass
from typing import Self

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence

from learning.learner import Learner
from learning.names_generator.names_generator_dataset import NameSample


class NamesGenerator(nn.Module):
    def __init__(self, hidden_size, num_vocab, num_classes):
        super().__init__()
        self.hidden_size = hidden_size

        # [S, C + V + H] -> [S, H]
        self.i2h = nn.Linear(num_classes + num_vocab + hidden_size, hidden_size)
        # [S, C + V + H] -> [S, V]
        self.i2o = nn.Linear(num_classes + num_vocab + hidden_size, num_vocab)
        # [S, H + V] -> [S, V]
        self.o2o = nn.Linear(hidden_size + num_vocab, num_vocab)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        """
        category: [S, C]
        input: [S, V]
        hidden: [S, H]
        """
        # print(f"{category.shape=}, {input.shape=}, {hidden.shape=}")

        # [S, C + V + H]
        input_combined = torch.cat((category, input, hidden), 1)
        # print(f"{input_combined.shape=}")

        # [S, H]
        hidden = self.i2h(input_combined)
        # print(f"{hidden.shape=}")

        # [S, V]
        output = self.i2o(input_combined)
        # print(f"{output.shape=}")

        # [S, H + V]
        output_combined = torch.cat((hidden, output), 1)
        # print(f"{output_combined.shape=}")

        # [S, V]
        output = self.o2o(output_combined)
        # print(f"{output.shape=}")

        # [S, V]
        output = self.dropout(output)
        # [S, V]
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)


@dataclass
class Batch:
    # list[torch.Tensor] -- list of [S, C]
    categories: list[torch.Tensor]
    # list[torch.Tensor] -- list of [S, V]
    inputs: list[torch.Tensor]
    # shape: [N, S, V] -- label must be in tensor form
    labels: torch.Tensor

    @classmethod
    def from_samples(cls, batch: list[NameSample]) -> Self:
        categories = []
        inputs = []
        labels = []
        for sample in batch:
            categories.append(sample.category)
            inputs.append(sample.input)
            labels.append(sample.label)

        labels_tensor = torch.cat(labels)
        return cls(categories=categories, inputs=inputs, labels=labels_tensor)


class SequentialBatchLearner(Learner[Batch]):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, int]:
        outputs: list[torch.Tensor] = []
        batch_loss = torch.tensor(0.0)

        # batch.inputs: list[torch.Tensor] -- list of [S, D]
        # batch.labels: [N, S, V]
        for input, label in zip(batch.inputs, batch.labels):
            # output: [1, C]
            output = self.model(input)
            outputs.append(output)
            batch_loss += self.criterion(output, label.unsqueeze(0))

        # outputs: [N, C]
        outputs_tensor = torch.cat(outputs, dim=0)

        return outputs_tensor, batch_loss, len(batch.inputs)


class ParallelBatchLearner(Learner):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, int]:
        # inputs: [S, N, D]
        # batch.labels: [N]
        inputs = pad_sequence(batch.inputs).squeeze(2)
        # shape: [N, C]
        outputs = self.model(inputs)
        batch_loss = self.criterion(outputs, batch.labels)
        return outputs, batch_loss, 1
