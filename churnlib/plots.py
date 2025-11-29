from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve


def plot_roc_curve(y_true, y_proba, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a ROC curve given true labels and predicted probabilities."""
    if ax is None:
        ax = plt.gca()
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    return ax


def plot_pr_curve(y_true, y_proba, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """Plot a precision–recall curve."""
    if ax is None:
        ax = plt.gca()
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax.plot(recall, precision)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve")
    return ax
