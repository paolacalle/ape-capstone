
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
)
from sklearn.model_selection import train_test_split
from scipy import stats


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