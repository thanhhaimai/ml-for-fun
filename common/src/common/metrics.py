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

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        batch_loss: torch.Tensor,
    ):
        """
        batch_loss: [1]
        """
        if batch_loss.shape != (1,):
            raise ValueError(
                f"Expect batch_loss.shape == (1,), but got {batch_loss.shape}"
            )

        self.loss += batch_loss.item()

    def on_epoch_complete(self, epoch_idx: int):
        self.loss /= self.batch_size
        self.epoch_losses[epoch_idx] = self.loss


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
        self.correct = 0

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
        self.correct += (predictions == labels).sum(0).item()

    def on_epoch_complete(self, epoch_idx: int):
        self.correct /= self.batch_size
        self.epoch_corrects[epoch_idx] = self.correct


class ConfusionMatrixMetric(Metric):
    def __init__(self, num_epochs: int, batch_size: int, num_classes: int):
        """
        num_epochs: E
        batch_size: N
        num_classes: C
        """
        super().__init__(num_epochs, batch_size)
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(num_classes, num_classes)

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
        for i in range(self.batch_size):
            print(f"{i=} {labels[i]=}, {predictions[i]=}")
            self.confusion_matrix[labels[i], predictions[i]] += 1

    def on_epoch_complete(self, epoch_idx: int):
        # Track the confusion matrix for the whole training
        pass
