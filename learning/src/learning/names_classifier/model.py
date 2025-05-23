from dataclasses import dataclass
from typing import Self

import torch
from torch import nn, optim
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_sequence

from learning.learner import BatchResult, Learner
from learning.names_classifier.names_classifier_dataset import NameSample


class NamesClassifierRNN(nn.Module):
    """
    D: input_size
    H: hidden_size
    C: output_size

    S: sequence_length
    N: batch_size
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # rnn: [S, N, D] -> hidden [N, H]
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)

        # fc: [N, H] -> [N, C]
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [S, N, D]
        """
        # hidden: [N, H]
        _rnn_output, hidden = self.rnn(x)

        # output: [N, C]
        output = self.fc(hidden[0])
        return output


class NamesClassifierLSTM(nn.Module):
    """
    D: input_size
    H: hidden_size
    C: output_size

    S: sequence_length
    N: batch_size
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # num_layers * num_directions == 4
        # lstm: [S, N, D] -> hidden [4, N, H]
        # Need to concatenate the last 2 hidden states since this is a bidirectional LSTM
        # hidden: [4, N, H] -> [N, H * 2]
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=False,
            num_layers=1,
            bidirectional=True,
        )

        # dropout: [N, H * 2] -> [N, H * 2]
        # self.dropout = nn.Dropout(p=0.1)

        # fc: [N, H * 2] -> [N, C]
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [S, N, D]
        """
        # hidden: [num_layers * num_directions, N, H]
        _lstm_output, (hidden, _cell) = self.lstm(x)

        # bidirectional_hidden_state: [N, H * 2]
        bidirectional_hidden_state = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # dropout_output: [N, H * 2]
        # dropout_output = self.dropout(bidirectional_hidden_state)

        # output: [N, C]
        output = self.fc(bidirectional_hidden_state)
        return output


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
        # Sort samples by input length in descending order
        batch.samples.sort(key=lambda x: len(x.input), reverse=True)

        # sample.input: [S, 1, V]
        # inputs: [S, N, V]
        padded_inputs = pad_sequence(
            [sample.input for sample in batch.samples],
            batch_first=False,
        ).squeeze(2)

        packed_inputs: PackedSequence = pack_padded_sequence(
            input=padded_inputs,
            lengths=[len(sample.input) for sample in batch.samples],
            batch_first=False,
            enforce_sorted=True,
        )

        # sample.label: [1]
        # labels: [N]
        labels = torch.cat([sample.label for sample in batch.samples])

        # shape: [N, C]
        outputs = self.model(packed_inputs)
        batch_loss = self.criterion(outputs, labels)

        return BatchResult(
            outputs=outputs,
            labels=labels,
            loss=batch_loss,
            loss_scale=len(batch.samples),
        )
