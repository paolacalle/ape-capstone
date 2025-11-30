# utils/plot_utils.py

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

COLORS = [
    "blue",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "cyan",
]

TEXT = {
    "title_fontsize": 16,
    "label_fontsize": 14,
    "tick_fontsize": 12,
    "legend_fontsize": 12,
}


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_distribution(
    df: pd.DataFrame,
    column: str,
    bins: int = 30,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "Count",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Histogram for a single numeric column."""
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=bins, edgecolor="black")
    ax.set_title(title or f"Distribution of {column}")
    ax.set_xlabel(xlabel or column)
    ax.set_ylabel(ylabel)

    if output_path is not None:
        output_path = ensure_dir(Path(output_path).parent) / Path(output_path).name
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def plot_group_boxplot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[Sequence] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Boxplot of a numeric variable across groups (e.g., rating by gender).
    `group_col` could be a categorical column or a derived label.
    """
    data = df.copy()

    if groups is not None:
        data = data[data[group_col].isin(groups)]

    fig, ax = plt.subplots()
    data.boxplot(column=value_col, by=group_col, ax=ax)
    ax.set_title(title or f"{value_col} by {group_col}")
    ax.set_ylabel(ylabel or value_col)
    plt.suptitle("")  # remove automatic suptitle

    if output_path is not None:
        output_path = ensure_dir(Path(output_path).parent) / Path(output_path).name
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def plot_group_violin(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    groups: Optional[Sequence] = None,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """
    Violin-like plot using matplotlib (not seaborn) for distributions per group.
    """
    data = df.copy()
    if groups is not None:
        data = data[data[group_col].isin(groups)]

    group_values = []
    group_labels = []
    for g, sub in data.groupby(group_col):
        group_values.append(sub[value_col].dropna().values)
        group_labels.append(g)

    fig, ax = plt.subplots()
    ax.violinplot(group_values, showmeans=True, showmedians=True)
    ax.set_xticks(np.arange(1, len(group_labels) + 1))
    ax.set_xticklabels(group_labels)
    ax.set_title(title or f"{value_col} by {group_col}")
    ax.set_ylabel(ylabel or value_col)

    if output_path is not None:
        output_path = ensure_dir(Path(output_path).parent) / Path(output_path).name
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def plot_regression_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Basic residual plot: y_true vs residuals."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual")
    ax.set_title(title)

    if output_path is not None:
        output_path = ensure_dir(Path(output_path).parent) / Path(output_path).name
        fig.savefig(output_path, bbox_inches="tight")

    return fig


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_value: Optional[float] = None,
    title: str = "ROC Curve",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot ROC curve given fpr/tpr and optional AUROC."""
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--")
    label = "ROC"
    if auc_value is not None:
        label += f" (AUROC = {auc_value:.3f})"
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    if output_path is not None:
        output_path = ensure_dir(Path(output_path).parent) / Path(output_path).name
        fig.savefig(output_path, bbox_inches="tight")

    return fig
