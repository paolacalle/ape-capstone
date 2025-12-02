import numpy as np
import pytest

from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    roc_curve as sk_roc_curve,
    roc_auc_score,
)

from utils.model_scores import Scores
from utils.seed_setter import SeedSetter
SEED = SeedSetter.get_seed()

def test_confusion_matrix_matches_sklearn():
    # simple, small test case
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])

    tp, fp, tn, fn = Scores.confusion_matrix(y_true, y_pred)

    # sklearn confusion_matrix returns:
    # [[tn, fp],
    #  [fn, tp]]
    sk_cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
    sk_tn, sk_fp, sk_fn, sk_tp = sk_cm.ravel()

    assert tp == sk_tp
    assert fp == sk_fp
    assert tn == sk_tn
    assert fn == sk_fn


def test_basic_metrics_match_sklearn():
    rng = np.random.default_rng(SEED)
    y_true = rng.integers(0, 2, size=200)
    y_pred = rng.integers(0, 2, size=200)

    # Accuracy
    my_acc = Scores.accuracy(y_true, y_pred)
    sk_acc = accuracy_score(y_true, y_pred)
    assert np.isclose(my_acc, sk_acc)

    # Sensitivity = recall for positive class
    my_sens = Scores.sensitivity(y_true, y_pred)
    sk_recall = recall_score(y_true, y_pred, pos_label=1)
    if not np.isnan(my_sens):
        assert np.isclose(my_sens, sk_recall)

    # Specificity = recall for negative class
    my_spec = Scores.specificity(y_true, y_pred)
    sk_spec = recall_score(y_true, y_pred, pos_label=0)
    if not np.isnan(my_spec):
        assert np.isclose(my_spec, sk_spec)

    # Precision = precision for positive class
    my_prec = Scores.precision(y_true, y_pred)
    sk_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    # When sklearn returns 0 due to zero_division and you return nan,
    # we just skip the check. Otherwise compare.
    if not np.isnan(my_prec):
        assert np.isclose(my_prec, sk_prec)


def test_prevalence_is_fraction_of_positives():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    prev = Scores.prevalence(y_true)

    expected = np.sum(y_true == 1) / len(y_true)
    assert np.isclose(prev, expected)

def test_evaluate_dispatches_correctly():
    rng = np.random.default_rng(SEED)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.integers(0, 2, size=100)
    y_scores = rng.random(size=100)

    methods = ["accuracy", "sensitivity", "specificity", "precision", "npv", "auc_scratch", "roc_scratch", "auc", "roc"]
    results = Scores.evaluate(y_true, y_pred, y_scores, methods)

    # Check keys
    assert set(results.keys()) == set(methods)

    # Spot-check types
    assert isinstance(results["accuracy"], float)
    assert isinstance(results["auc"], float)

    # roc should be a tuple of (fpr, tpr, thresholds)
    roc_res = results["roc"]
    assert isinstance(roc_res, tuple)
    assert len(roc_res) == 3
    fpr, tpr, thresh = roc_res
    assert len(fpr) == len(tpr) == len(thresh)
