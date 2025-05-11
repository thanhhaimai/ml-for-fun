import pytest
import torch

from common.metrics import ConfusionMatrixMetric


def test_confusion_matrix_update_and_properties():
    # 2-class problem: classes 0 and 1
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)

    # Simulate a batch: 10 samples, 2 classes
    # outputs: logits for 2 classes
    # We want the confusion matrix to be [[1, 2], [3, 4]]
    # That means:
    # - True 0, Pred 0: 1
    # - True 0, Pred 1: 2
    # - True 1, Pred 0: 3
    # - True 1, Pred 1: 4
    outputs = torch.tensor(
        [
            [2.0, 1.0],  # pred: 0, label: 0 (1)
            [1.0, 2.0],  # pred: 1, label: 0 (2)
            [1.0, 2.0],  # pred: 1, label: 0 (2)
            [2.0, 1.0],  # pred: 0, label: 1 (3)
            [2.0, 1.0],  # pred: 0, label: 1 (3)
            [2.0, 1.0],  # pred: 0, label: 1 (3)
            [1.0, 2.0],  # pred: 1, label: 1 (4)
            [1.0, 2.0],  # pred: 1, label: 1 (4)
            [1.0, 2.0],  # pred: 1, label: 1 (4)
            [1.0, 2.0],  # pred: 1, label: 1 (4)
        ]
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

    metric.update(outputs, labels)
    metric.on_epoch_complete(0)

    # The confusion matrix should be:
    # [[1, 2],
    #  [3, 4]]
    expected = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(metric.epoch_confusion_matrices[0], expected)

    # Accuracy: (1+4)/10 = 0.5
    assert torch.isclose(metric.accuracies[0], torch.tensor(0.5))

    # Precision for class 0: 1/(1+3) = 0.25
    # Precision for class 1: 4/(2+4) = 0.666...
    precisions = metric.class_precisions
    assert torch.allclose(precisions, torch.tensor([0.25, 0.6666667]), atol=1e-5)

    # Recall for class 0: 1/(1+2) = 0.333...
    # Recall for class 1: 4/(3+4) = 0.571...
    recalls = metric.class_recalls
    assert torch.allclose(recalls, torch.tensor([0.3333333, 0.5714286]), atol=1e-5)

    # F1 for class 0: 2*0.25*0.333.../(0.25+0.333...) = 0.2857
    # F1 for class 1: 2*0.666...*0.571.../(0.666...+0.571...) = 0.6154
    f1s = metric.class_f1_scores
    assert torch.allclose(f1s, torch.tensor([0.2857143, 0.6153846]), atol=1e-5)


def test_confusion_matrix_reset():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)
    outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1])
    metric.update(outputs, labels)
    metric.reset_confusion_matrix()
    assert torch.equal(metric.confusion_matrix, torch.zeros(2, 2, dtype=torch.int64))


def test_confusion_matrix_shape_and_error():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)

    outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels = torch.tensor([0])  # Wrong shape
    with pytest.raises(ValueError):
        metric.update(outputs, labels)

    outputs = torch.tensor([[2.0, 1.0, 0.0], [1.0, 2.0, 0.0]])  # Wrong num classes
    labels = torch.tensor([0, 1])
    with pytest.raises(ValueError):
        metric.update(outputs, labels)


def test_multiple_epochs():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)
    # Epoch 1
    outputs1 = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels1 = torch.tensor([0, 1])
    metric.update(outputs1, labels1)
    metric.on_epoch_complete(0)
    # Epoch 2
    outputs2 = torch.tensor([[1.0, 2.0], [2.0, 1.0]])
    labels2 = torch.tensor([1, 0])
    metric.update(outputs2, labels2)
    metric.on_epoch_complete(1)
    # Should have 2 confusion matrices
    assert len(metric.epoch_confusion_matrices) == 2
    # Accuracies should be [1.0, 1.0]
    assert torch.allclose(metric.accuracies, torch.tensor([1.0, 1.0]), atol=1e-5)


def test_single_class():
    classes = ["A"]
    metric = ConfusionMatrixMetric(classes=classes)
    outputs = torch.tensor([[1.0]])
    labels = torch.tensor([0])
    metric.update(outputs, labels)
    metric.on_epoch_complete(0)
    expected = torch.tensor([[1]])
    assert torch.equal(metric.epoch_confusion_matrices[0], expected)
    assert torch.isclose(metric.accuracies[0], torch.tensor(1.0))
    assert torch.allclose(metric.class_precisions, torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(metric.class_recalls, torch.tensor([1.0]), atol=1e-5)
    assert torch.allclose(metric.class_f1_scores, torch.tensor([1.0]), atol=1e-5)


def test_empty_update():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)
    outputs = torch.empty((0, 2))
    labels = torch.empty((0,), dtype=torch.long)
    # Should not raise
    metric.update(outputs, labels)
    metric.on_epoch_complete(0)
    expected = torch.zeros(2, 2, dtype=torch.int64)
    assert torch.equal(metric.epoch_confusion_matrices[0], expected)
    assert torch.isnan(metric.accuracies[0]) or metric.accuracies[0] == 0.0


def test_invalid_labels():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)
    outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels = torch.tensor([0, -1])  # Negative label
    with pytest.raises(IndexError):
        metric.update(outputs, labels)
    labels = torch.tensor([0, 2])  # Out of bounds
    with pytest.raises(IndexError):
        metric.update(outputs, labels)
    # Float labels
    labels = torch.tensor([0.0, 1.0])
    with pytest.raises(RuntimeError):
        metric.update(outputs, labels)


def test_eps_handling():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes, eps=1.0)
    outputs = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1])
    metric.update(outputs, labels)
    metric.on_epoch_complete(0)
    # With eps=1.0, precision/recall/f1 should be lower
    assert torch.all(metric.class_precisions < 1.0)
    assert torch.all(metric.class_recalls < 1.0)
    assert torch.all(metric.class_f1_scores < 1.0)


def test_properties_before_epoch():
    classes = ["A", "B"]
    metric = ConfusionMatrixMetric(classes=classes)
    # Should raise IndexError if properties accessed before any epoch
    with pytest.raises(IndexError):
        _ = metric.class_precisions
    with pytest.raises(IndexError):
        _ = metric.class_recalls
    with pytest.raises(IndexError):
        _ = metric.class_f1_scores
    # accuracies property should also fail
    with pytest.raises(RuntimeError):
        _ = metric.accuracies
