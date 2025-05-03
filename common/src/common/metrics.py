import torch


class Metric:
    def __init__(self, batch_size: int):
        """
        batch_size: N
        """
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


class AccuracyMetric(Metric):
    def __init__(self, batch_size: int, num_classes: int):
        """
        batch_size: N
        num_classes: C
        """
        super().__init__(batch_size)
        self.num_classes = num_classes
        self.epoch_corrects = []
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
        self.epoch_corrects.append(self.total_correct / self.total_items)
        self.total_correct = 0
        self.total_items = 0


class ConfusionMatrixMetric(Metric):
    def __init__(self, batch_size: int, num_classes: int):
        """
        batch_size: N
        num_classes: C
        """
        super().__init__(batch_size)
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
        indices = labels * self.num_classes + predictions
        self.confusion_matrix.view(-1).index_add_(0, indices, torch.ones_like(indices))

    def on_epoch_complete(self, epoch_idx: int):
        # Track the confusion matrix for the whole training
        pass
