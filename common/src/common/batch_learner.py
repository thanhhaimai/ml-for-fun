import torch
from torch import nn, optim

from common.learner import Batch, Learner


class SequentialBatchLearner(Learner):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        outputs: list[torch.Tensor] = []
        batch_loss = torch.tensor(0.0)

        # inputs: [N, S, 1, D]
        # labels: [N]
        # input: [S, 1, D]
        # label: scalar
        for input, label in zip(batch.inputs, batch.labels):
            # output: [1, C]
            output = self.model(input)
            outputs.append(output)
            loss = self.criterion(output, label.unsqueeze(0))
            batch_loss += loss

        # outputs: [N, C]
        outputs_tensor = torch.cat(outputs, dim=0)

        return outputs_tensor, batch_loss


class ParallelBatchLearner(Learner):
    def __init__(
        self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module
    ):
        super().__init__(model, optimizer, criterion)

    def batch_step(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        # batch.inputs: [N, S, 1, D]
        # batch.labels: [N]
        # outputs: [N, C]
        outputs = self.model(batch.inputs)
        batch_loss = self.criterion(outputs, batch.labels)
        return outputs, batch_loss
