import numpy as np
import pandas as pd
import pytest
from utils.seed_setter import SeedSetter
from utils.modeling_helpers import ModelHelper
from sklearn.linear_model import LogisticRegression
SEED = SeedSetter.get_seed()


def test_model_helper_cross_val_with_logistic_regression():
    rng = np.random.default_rng(seed=SEED)
    X = rng.normal(size=(200, 3))
    y = (rng.random(200) > 0.5).astype(int)

    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["y"] = y

    helper = ModelHelper(df)
    kf = helper.kfold_split(n_splits=4)

    model = LogisticRegression(max_iter=1000)
    scoring_methods = ["accuracy", "auc"]

    fold_scores = helper.cross_val_score(
        model,
        df[["x1", "x2", "x3"]],
        df["y"],
        kf,
        scoring_methods=scoring_methods,
    )

    assert set(fold_scores.keys()) == set(scoring_methods)
    assert len(fold_scores["accuracy"]) == 4
    assert len(fold_scores["auc"]) == 4
