import torch
from torch import nn, optim
from torch.nn.utils.rnn import pad_sequence

from common.learner import Batch, Learner, Sample


class SequentialBatchLearner(Learner):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def collate_batch(self, batch: list[Sample]) -> Batch:
        """
        batch: list[Sample]
        """
        inputs = []
        labels = []
        for sample in batch:
            inputs.append(sample.input)
            labels.append(sample.label)

        labels_tensor = torch.cat(labels)
        return Batch(inputs=inputs, labels=labels_tensor)

    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, int]:
        outputs: list[torch.Tensor] = []
        batch_loss = torch.tensor(0.0)

        # inputs: list[torch.Tensor] -- [S, 1, D]
        # labels: [N]
        # input: [S, 1, D]
        # label: scalar
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

    def collate_batch(self, batch: list[Sample]) -> Batch:
        """
        batch: list[Sample]
        """
        inputs = []
        labels = []
        for sample in batch:
            inputs.append(sample.input)
            labels.append(sample.label)

        labels_tensor = torch.cat(labels)
        return Batch(inputs=inputs, labels=labels_tensor)

    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, int]:
        # inputs: [S, N, D]
        # batch.labels: [N]
        inputs = pad_sequence(batch.inputs).squeeze(2)
        # shape: [N, C]
        outputs = self.model(inputs)
        batch_loss = self.criterion(outputs, batch.labels)
        return outputs, batch_loss, 1
