
# This will hold generic stats + models you can 
# reuse across Q1–Q10 without actually plugging in specific columns yet.

from dataclasses import dataclass
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
    ):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        return kf
    
    def cross_val_score(
        self,
        model, 
        x, 
        y,
        kfolds,
        scoring=[]
    ):
        fold_results = {
            
        }
        for train_index, test_index in kfolds.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            