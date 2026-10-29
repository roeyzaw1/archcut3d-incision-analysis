import numpy as np

from archcut3d.ml.train import stop_metrics


def test_perfect_separation():
    m = stop_metrics(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0]))
    assert m["auroc"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    assert m["accuracy"] == 1.0 and m["base_rate"] == 0.5


def test_never_stop_has_high_accuracy_but_zero_recall_and_chance_auroc():
    target = np.zeros(100)
    target[:3] = 1                                   # 3% positives
    m = stop_metrics(np.full(100, 0.01), target)
    assert abs(m["accuracy"] - 0.97) < 1e-9          # accuracy looks great
    assert m["recall"] == 0.0
    assert np.isnan(m["precision"])                  # no predicted positives
    assert m["auroc"] == 0.5                         # constant score = chance
    assert abs(m["base_rate"] - 0.03) < 1e-9


def test_undefined_when_one_class_only():
    m = stop_metrics(np.array([0.1, 0.2]), np.array([0, 0]))
    assert np.isnan(m["auroc"]) and np.isnan(m["recall"]) and m["base_rate"] == 0.0


def test_threshold_matters_for_precision_recall_not_auroc():
    prob = np.array([0.6, 0.4, 0.3, 0.2])
    target = np.array([1, 1, 0, 0])
    lo = stop_metrics(prob, target, threshold=0.35)
    hi = stop_metrics(prob, target, threshold=0.5)
    assert lo["auroc"] == hi["auroc"] == 1.0
    assert lo["recall"] == 1.0 and hi["recall"] == 0.5
