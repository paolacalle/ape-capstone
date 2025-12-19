
# This will hold generic stats + models you can 
# reuse across Q1–Q10 without actually plugging in specific columns yet.

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import ClassifierMixin, RegressorMixin, clone
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
    precision_score, 
    confusion_matrix
)

from sklearn.model_selection import (
    train_test_split as tt,
    KFold, 
    cross_val_score
)
from scipy import stats
from .seed_setter import SeedSetter
from .model_scores import Scores


# ---------- Dataclasses for clean returns ---------- #

@dataclass
class GroupComparisonResult:
    group1: str
    group2: str
    mean1: float
    mean2: float
    diff: float
    p_value: float
    test_name: str


@dataclass
class VarianceComparisonResult:
    group1: str
    group2: str
    var1: float
    var2: float
    ratio: float
    p_value: float
    test_name: str


@dataclass
class RegressionResult:
    model: LinearRegression
    feature_names: List[str]
    coefficients: np.ndarray
    intercept: float
    r2: float
    rmse: float


@dataclass
class ClassificationResult:
    model: LogisticRegression
    feature_names: List[str]
    auc: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


# --------- Will define methods for modeling and evaluation --------- #
class ModelHelper:
    def __init__(
        self, 
        df: pd.DataFrame,
        random_seed: int = None
    ):
        # if random_seed is None, get seed from .env
        if random_seed is None:
            random_seed = SeedSetter.set_seed(random_seed)
        self.random_seed = random_seed
        self.df = df
        
    # -- splitter --- 
    def train_test_split(
        self,
        test_size: float = 0.2,
    ): 
        train, test = tt(self.df, test_size=test_size, random_state=self.random_seed)
        return train, test
    
    def kfold_split(
        self,
        n_splits: int = 5,
        shuffle: bool = True
    ) -> KFold:
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_seed)
        return kf
    
    def cross_val_score(
        self,
        model, 
        X,
        y,
        kfolds,
        scoring_methods: Optional[List[str]] = None
    ) -> Dict[str, Any]:

        if scoring_methods is None:
            scoring_methods = []

        Scores.verify_scoring_methods(scoring_methods)

        is_pandas_X = hasattr(X, "iloc")
        is_pandas_y = hasattr(y, "iloc")

        X_data = X
        y_data = y

        is_classifier = isinstance(model, ClassifierMixin)
        is_regressor = isinstance(model, RegressorMixin)

        if not (is_classifier or is_regressor):
            raise ValueError("Model must be a classifier or regressor.")

        fold_results = defaultdict(list)
        fitted_models = []

        for train_idx, test_idx in kfolds.split(X_data):

            # fresh unfitted estimator each fold
            f_model = clone(model)

            X_train = X_data.iloc[train_idx] if is_pandas_X else X_data[train_idx]
            X_test  = X_data.iloc[test_idx]  if is_pandas_X else X_data[test_idx]

            y_train = y_data.iloc[train_idx] if is_pandas_y else y_data[train_idx]
            y_test  = y_data.iloc[test_idx]  if is_pandas_y else y_data[test_idx]

            f_model.fit(X_train, y_train)

            y_pred = f_model.predict(X_test)

            # classification vs regression
            if is_classifier:
                if hasattr(f_model, "predict_proba"):
                    y_pred_scores = f_model.predict_proba(X_test)[:, 1]
                elif hasattr(f_model, "decision_function"):
                    y_pred_scores = f_model.decision_function(X_test)
                else:
                    y_pred_scores = None
            else:
                y_pred_scores = None

            res = Scores.evaluate(
                y_true=y_test,
                y_pred=y_pred,
                y_pred_scores=y_pred_scores,
                methods=scoring_methods
            )

            for method, score in res.items():
                fold_results[method].append(score)

            fitted_models.append(f_model)

        # store the list (not the last model)
        fold_results["fitted_models"] = fitted_models

        return dict(fold_results)

