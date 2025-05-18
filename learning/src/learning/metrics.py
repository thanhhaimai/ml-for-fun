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
    ):
        """
        outputs: [N, C]
        labels: [N]
        """
        raise NotImplementedError

    def on_epoch_complete(self, epoch_idx: int):
        pass


class ConfusionMatrixMetric(Metric):
    def __init__(self, classes: list[str], eps: float = 1e-6):
        """
        num_classes: C
        """
        super().__init__()
        self.classes = classes
        self.num_classes = len(classes)
        self.eps = eps

        # All confusion matrices across epochs
        self.epoch_confusion_matrices: list[torch.Tensor] = []
        self.reset_confusion_matrix()

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        outputs: [*, N, C] -- batch of logits
        labels: [*, N] -- batch of label indices
        """
        if outputs.shape[-1] != self.num_classes:
            raise ValueError(
                f"Expect outputs.shape[-1] == {self.num_classes}, but got {outputs.shape[-1]}"
            )

        if outputs.shape[-2] != labels.shape[-1]:
            raise ValueError(
                f"Expect outputs.shape[-2] == labels.shape[-1], but got {outputs.shape[-2]} != {labels.shape[-1]}"
            )

        outputs = outputs.view(-1, self.num_classes)
        labels = labels.view(-1)

        # predictions: [N]
        predictions = outputs.argmax(dim=1)
        indices = labels * self.num_classes + predictions
        self.confusion_matrix.view(-1).index_add_(0, indices, torch.ones_like(indices))

    def on_epoch_complete(self, epoch_idx: int):
        # Track the confusion matrix for the whole training
        self.epoch_confusion_matrices.append(self.confusion_matrix)
        self.reset_confusion_matrix()

    def reset_confusion_matrix(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes,
            self.num_classes,
            dtype=torch.int64,
        )

    @property
    def accuracies(self):
        """
        Accuracies for all epochs

        E: number of epochs
        C: number of classes
        """
        # shape: [E, C, C]
        matrices = torch.stack(self.epoch_confusion_matrices, dim=0)
        # shape: [E, C]
        diagonals = matrices.diagonal(dim1=-2, dim2=-1)
        # shape: [E] / [E] = [E]
        return diagonals.sum(dim=-1) / matrices.sum(dim=-1).sum(dim=-1)

    @property
    def class_precisions(self):
        """
        Precisions for all classes

        C: number of classes
        """
        # shape: [C, C]
        matrix = self.epoch_confusion_matrices[-1]
        # shape: [C] / [C] = [C]
        return matrix.diag() / (matrix.sum(dim=0) + self.eps)

    @property
    def class_recalls(self):
        """
        Recalls for all classes

        C: number of classes
        """
        # shape: [C, C]
        matrix = self.epoch_confusion_matrices[-1]
        # shape: [C] / [C] = [C]
        return matrix.diag() / (matrix.sum(dim=1) + self.eps)

    @property
    def class_f1_scores(self):
        """
        F1 scores for all classes

        C: number of classes
        """
        # shape: [C]
        precisions = self.class_precisions
        # shape: [C]
        recalls = self.class_recalls
        # shape: [C] * [C] / [C] = [C]
        return 2 * (precisions * recalls) / (precisions + recalls + self.eps)

    def plot_confusion_matrix(self, ax: matplotlib.axes.Axes, normalize: bool = False):
        if normalize:
            matrix = self.epoch_confusion_matrices[
                -1
            ].float() / self.epoch_confusion_matrices[-1].sum(dim=1, keepdim=True)
            format = ".2f"
        else:
            matrix = self.epoch_confusion_matrices[-1]
            format = "d"

        sns.heatmap(
            data=matrix,
            annot=True,
            fmt=format,
            ax=ax,
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Label")
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        ax.set_yticklabels(self.classes, rotation=0)

    def plot_accuracies(self, ax: matplotlib.axes.Axes, label: str):
        ax.plot(self.accuracies, label=label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()

    def plot_class_precisions(self, ax: matplotlib.axes.Axes, label: str):
        ax.plot(self.class_precisions, marker="o", label=label)
        ax.set_xlabel("Class")
        ax.set_ylabel("Precision")
        ax.set_xticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        ax.legend()

    def plot_class_recalls(self, ax: matplotlib.axes.Axes, label: str):
        ax.plot(self.class_recalls, marker="o", label=label)
        ax.set_xlabel("Class")
        ax.set_ylabel("Recall")
        ax.set_xticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        ax.legend()

    def plot_class_f1_scores(self, ax: matplotlib.axes.Axes, label: str):
        ax.plot(self.class_f1_scores, marker="o", label=label)
        ax.set_xlabel("Class")
        ax.set_ylabel("F1 Score")
        ax.set_xticks(range(len(self.classes)))
        ax.set_xticklabels(self.classes, rotation=45, ha="right")
        ax.legend()
