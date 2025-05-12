from dataclasses import dataclass
from typing import Self

import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence

from learning.learner import BatchResult, Learner
from learning.names_generator.names_generator_dataset import NameSample


class NamesGenerator(nn.Module):
    def __init__(self, hidden_size, num_vocab, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_vocab = num_vocab
        self.num_classes = num_classes

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
        category: [1, C]
        input: [S, V]
        hidden: [1, H]
        """
        # print(f"{category.shape=}, {input.shape=}, {hidden.shape=}")
        sequence_length = input.shape[0]

        # [S, C + V + H]
        input_combined = torch.cat(
            (
                category.expand(sequence_length, -1),
                input,
                hidden.expand(sequence_length, -1),
            ),
            1,
        )
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
        batch_loss = torch.tensor(0.0)
        loss_count = 0

        # Declare that the model is a NamesGenerator
        self.model: NamesGenerator

        for sample in batch.samples:
            # For every sample, we need to initialize the hidden state from scratch
            # shape: [1, H]
            hidden = self.model.init_hidden()

            for x, y in zip(sample.input, sample.label):
                # output: [1, V]
                # hidden: [1, H]
                output, hidden = self.model(sample.category, x.unsqueeze(0), hidden)
                batch_loss += self.criterion(output, y.unsqueeze(0))
                loss_count += 1

        return BatchResult(
            outputs=torch.zeros([1, 1]),
            labels=torch.zeros([1, 1]),
            loss=batch_loss,
            loss_scale=loss_count,
        )


class ParallelBatchLearner(Learner):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> BatchResult:
        # shape: [N, S, V]
        inputs = pad_sequence([sample.input for sample in batch.samples])
        # shape: [N, S, 1]
        labels = pad_sequence([sample.label for sample in batch.samples])
        # shape: [N, 1, C]
        categories = pad_sequence([sample.category for sample in batch.samples])

        # shape: [N, S, C]
        outputs = self.model(categories, inputs)
        batch_loss = self.criterion(outputs, labels)

        return BatchResult(
            outputs=outputs,
            labels=labels,
            loss=batch_loss,
            loss_scale=1,
        )
