
# This will hold generic stats + models you can 
# reuse across Q1–Q10 without actually plugging in specific columns yet.

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
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
        self.random_seed = SeedSetter.set_seed(random_seed)
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
    ):
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=self.random_seed)
        return kf
    
    def cross_val_score(
        self,
        model, 
        X,
        y,
        kfolds: KFold,
        scoring_methods: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """
        Perform K-fold cross-validation and compute custom scores per fold.

        Parameters
        ----------
        model : sklearn-like estimator
            Must implement fit(X, y), predict(X), and predict_proba(X) for classification.
        X : array-like or DataFrame of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        kfolds : KFold
            A sklearn KFold (or similar) splitter.
        scoring_methods : list of str, optional
            Names of scoring methods as implemented in Scores. If None, no scores are computed.

        Returns
        -------
        dict
            {method_name: [score_fold_1, score_fold_2, ...]}
        """
        if scoring_methods is None:
            scoring_methods = []

        Scores.verify_scoring_methods(scoring_methods)

        # Ensure numpy array or proper indexer
        # This allows compatibility with both pandas DataFrames and numpy arrays
        if hasattr(X, "iloc"):
            X_data = X
            is_pandas_X = True
        else:
            X_data = np.asarray(X)
            is_pandas_X = False

        if hasattr(y, "iloc"):
            y_data = y
            is_pandas_y = True
        else:
            y_data = np.asarray(y)
            is_pandas_y = False

        fold_results: Dict[str, List[float]] = defaultdict(list)
        
        for train_index, test_index in kfolds.split(X_data):
            if is_pandas_X:
                X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
            else:
                X_train, X_test = X_data[train_index], X_data[test_index]

            if is_pandas_y:
                y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
            else:
                y_train, y_test = y_data[train_index], y_data[test_index]

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            # classification-only for now
            assert hasattr(model, "predict_proba"), "Model does not have predict_proba method"
            y_pred_scores = model.predict_proba(X_test)[:, 1]

            res = Scores.evaluate(y_test, y_pred, y_pred_scores, scoring_methods)
            
            for method, score in res.items():
                fold_results[method].append(score)
            
        return fold_results
    