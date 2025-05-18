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

    def forward(self, category, input, hidden):
        """
        category: [N, C]
        input: [N, V]
        hidden: [N, H]
        """
        # print(f"{category.shape=}, {input.shape=}, {hidden.shape=}")

        # [N, C + V + H]
        input_combined = torch.cat((category, input, hidden), -1)
        # print(f"{input_combined.shape=}")

        # [N, H]
        hidden = self.i2h(input_combined)
        # print(f"{hidden.shape=}")

        # [N, V]
        output = self.i2o(input_combined)
        # print(f"{output.shape=}")

        # [N, H + V]
        output_combined = torch.cat((hidden, output), -1)
        # print(f"{output_combined.shape=}")

        # [N, V]
        output = self.o2o(output_combined)
        # print(f"{output.shape=}")

        # [N, V]
        output = self.dropout(output)
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
        # Declare that the model is a NamesGenerator
        self.model: NamesGenerator

        batch_loss = torch.tensor(0.0)
        loss_count = 0
        for sample in batch.samples:
            # For every sample, we need to initialize the hidden state from scratch
            # shape: [1, H]
            hidden = self.model.init_hidden()

            # input: [1, V]
            # label: [1]
            for input, label in zip(sample.input, sample.label):
                # output: [1, V]
                # hidden: [1, H]
                output, hidden = self.model(sample.category, input.unsqueeze(0), hidden)

                batch_loss += self.criterion(output.squeeze(0), label)
                loss_count += 1

        return BatchResult(
            outputs=torch.zeros([1, 1]),
            labels=torch.zeros([1, 1]),
            loss=batch_loss,
            loss_scale=loss_count,
        )


class ParallelBatchLearner(Learner):
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
        # shape: [N, C]
        categories = torch.cat([sample.category for sample in batch.samples])

        # Declare that the model is a NamesGenerator
        self.model: NamesGenerator
        # shape: [N, H]
        hidden = self.model.init_hidden().expand(len(batch.samples), -1)

        # shape: [S, N, V]
        padded_inputs = pad_sequence([sample.input for sample in batch.samples])

        # shape: [S, N]
        padded_labels = pad_sequence(
            [sample.label for sample in batch.samples],
            batch_first=False,
            padding_value=self.padding_idx,
        )

        batch_loss = torch.tensor(0.0)
        total_valid_tokens = 0
        outputs = []

        # input: [N, V]
        # label: [N]
        for input, label in zip(padded_inputs, padded_labels):
            # output: [N, V]
            # hidden: [N, H]
            output, hidden = self.model(categories, input, hidden)
            outputs.append(output)

            num_valid_tokens = (label != self.padding_idx).sum().item()
            if num_valid_tokens == 0:
                continue

            loss = self.criterion(output, label)

            # Reduction is set to "sum"
            batch_loss += loss
            total_valid_tokens += num_valid_tokens

        return BatchResult(
            outputs=torch.stack(outputs),
            labels=padded_labels,
            loss=batch_loss,
            loss_scale=int(total_valid_tokens) if total_valid_tokens > 0 else 1,
        )
