import torch


class Metric:
    def __init__(self, num_epochs: int, batch_size: int):
        """
        num_epochs: E
        batch_size: N
        """
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        batch_loss: torch.Tensor,
    ):
        """
        outputs: [N, C]
        labels: [N]
        batch_loss: [1]
        """
        raise NotImplementedError

    def on_epoch_complete(self, epoch_idx: int):
        pass


class LossMetric(Metric):
    def __init__(self, num_epochs: int, batch_size: int):
        super().__init__(num_epochs, batch_size)
        self.epoch_losses = torch.zeros(num_epochs)
        self.loss = 0
        self.num_batches = 0

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        batch_loss: torch.Tensor,
    ):
        """
        batch_loss: scalar
        """
        self.loss += batch_loss.item()
        self.num_batches += 1

    def on_epoch_complete(self, epoch_idx: int):
        self.epoch_losses[epoch_idx] = self.loss / self.num_batches
        self.loss = 0
        self.num_batches = 0


class AccuracyMetric(Metric):
    def __init__(self, num_epochs: int, batch_size: int, num_classes: int):
        """
        num_epochs: E
        batch_size: N
        num_classes: C
        """
        super().__init__(num_epochs, batch_size)
        self.num_classes = num_classes
        self.epoch_corrects = torch.zeros(num_epochs)
        self.total_correct = 0
        self.total_items = 0

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        batch_loss: torch.Tensor,
    ):
        """
        outputs: [N, C]
        labels: [N]
        """
        if outputs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Expect outputs.shape[0] == labels.shape[0], but got {outputs.shape[0]} != {labels.shape[0]}"
            )

        if outputs.shape[1] != self.num_classes:
            raise ValueError(
                f"Expect outputs.shape[1] == {self.num_classes}, but got {outputs.shape[1]}"
            )

        # predictions: [N]
        predictions = outputs.argmax(dim=1)
        self.total_correct += (predictions == labels).sum(0).item()
        self.total_items += len(outputs)

    def on_epoch_complete(self, epoch_idx: int):
        self.epoch_corrects[epoch_idx] = self.total_correct / self.total_items
        self.total_correct = 0
        self.total_items = 0


class ConfusionMatrixMetric(Metric):
    def __init__(self, num_epochs: int, batch_size: int, num_classes: int):
        """
        num_epochs: E
        batch_size: N
        num_classes: C
        """
        super().__init__(num_epochs, batch_size)
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        batch_loss: torch.Tensor,
    ):
        """
        outputs: [N, C]
        labels: [N]
        """
        if outputs.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Expect outputs.shape[0] == labels.shape[0], but got {outputs.shape[0]} != {labels.shape[0]}"
            )

        if outputs.shape[1] != self.num_classes:
            raise ValueError(
                f"Expect outputs.shape[1] == {self.num_classes}, but got {outputs.shape[1]}"
            )

        # predictions: [N]
        predictions = outputs.argmax(dim=1)
        for i in range(len(labels)):
            u = int(labels[i].item())
            v = int(predictions[i].item())
            self.confusion_matrix[u, v] += 1

    def on_epoch_complete(self, epoch_idx: int):
        # Track the confusion matrix for the whole training
        pass
