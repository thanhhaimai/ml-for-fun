import matplotlib.axes
import seaborn as sns
import torch


class Metric:
    def __init__(self):
        """ """
        pass

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

    def plot(self, ax: matplotlib.axes.Axes, label: str | None = None):
        raise NotImplementedError


class AccuracyMetric(Metric):
    def __init__(self, num_classes: int):
        """
        num_classes: C
        """
        super().__init__()
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

    def plot(self, ax: matplotlib.axes.Axes, label: str | None = None):
        ax.plot(self.epoch_corrects, label=label)


class ConfusionMatrixMetric(Metric):
    def __init__(self, num_classes: int, normalize: bool):
        """
        num_classes: C
        """
        super().__init__()
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        self.normalize = normalize

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

    def plot(self, ax: matplotlib.axes.Axes, label: str | None = None):
        format = "d"
        if self.normalize:
            self.confusion_matrix = (
                self.confusion_matrix.float()
                / self.confusion_matrix.sum(dim=1, keepdim=True)
            )
            format = ".2f"

        sns.heatmap(
            data=self.confusion_matrix,
            annot=True,
            fmt=format,
            ax=ax,
        )

        ax.set_title("Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Label")


class PrecisionMetric(Metric):
    def __init__(self, num_classes: int):
        """
        Corrects / Predicteds (predictions == labels) / predictions

        batch_size: N
        num_classes: C
        """
        super().__init__()
        self.num_classes = num_classes
        self.true_positives = torch.zeros(num_classes, dtype=torch.int64)
        self.predicted_positives = torch.zeros(num_classes, dtype=torch.int64)
        self.epoch_precisions = []

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
        # is_correct: [N]
        is_correct_mask = predictions == labels
        # true_positive_labels: [K] -- where K is the number of true positives
        true_positive_labels = labels[is_correct_mask]
        # tp_counts (true positives): [C] -- since we set minlength=C
        tp_counts = torch.bincount(true_positive_labels, minlength=self.num_classes)
        # pp_counts (predicted positives): [C] -- since we set minlength=C
        pp_counts = torch.bincount(predictions, minlength=self.num_classes)

        self.true_positives += tp_counts
        self.predicted_positives += pp_counts

    def on_epoch_complete(self, epoch_idx: int):
        # precisions: [C]
        precisions = torch.zeros(self.num_classes, dtype=torch.float)

        # Create a mask for classes with predicted positives
        # Need a mask to avoid division by zero
        # has_pp_mask: [C]
        has_pp_mask = self.predicted_positives > 0

        # Calculate precision only for classes where pp > 0
        precisions[has_pp_mask] = (
            self.true_positives[has_pp_mask].float()
            / self.predicted_positives[has_pp_mask].float()
        )

        # Compute mean precision (macro average)
        mean_precision = precisions.mean().item()
        self.epoch_precisions.append(mean_precision)

        # Reset counts for the next epoch
        self.true_positives.zero_()
        self.predicted_positives.zero_()

    def plot(self, ax: matplotlib.axes.Axes, label: str | None = None):
        ax.plot(self.epoch_precisions, label=label)


class RecallMetric(Metric):
    def __init__(self, num_classes: int):
        """
        Corrects / Labels (predictions == labels / labels)

        batch_size: N
        num_classes: C
        """
        super().__init__()
        self.num_classes = num_classes
        self.true_positives = torch.zeros(num_classes, dtype=torch.int64)
        self.actual_positives = torch.zeros(num_classes, dtype=torch.int64)
        self.epoch_recalls = []

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
        # is_correct: [N]
        is_correct_mask = predictions == labels
        # true_positive_labels: [K] -- where K is the number of true positives
        true_positive_labels = labels[is_correct_mask]

        # tp_counts (true positives): [C]
        tp_counts = torch.bincount(true_positive_labels, minlength=self.num_classes)
        # ap_counts (actual positives): [C]
        ap_counts = torch.bincount(labels, minlength=self.num_classes)

        self.true_positives += tp_counts
        self.actual_positives += ap_counts

    def on_epoch_complete(self, epoch_idx: int):
        # recalls: [C]
        recalls = torch.zeros(self.num_classes, dtype=torch.float)

        # Create a mask for classes with actual positives
        # Need a mask to avoid division by zero
        # has_ap_mask: [C]
        has_ap_mask = self.actual_positives > 0

        # Calculate recall only for classes where cp > 0
        recalls[has_ap_mask] = (
            self.true_positives[has_ap_mask].float()
            / self.actual_positives[has_ap_mask].float()
        )

        # Compute mean recall (macro average)
        mean_recall = recalls.mean().item()
        self.epoch_recalls.append(mean_recall)

        # Reset counts for the next epoch
        self.true_positives.zero_()
        self.actual_positives.zero_()

    def plot(self, ax: matplotlib.axes.Axes, label: str | None = None):
        ax.plot(self.epoch_recalls, label=label)
