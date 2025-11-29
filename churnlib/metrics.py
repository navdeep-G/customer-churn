from typing import Dict, Hashable, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true,
    y_proba,
    positive_label: Hashable = 1,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute core binary classification metrics for churn models."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    y_hat = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat)),
        "recall": float(recall_score(y_true, y_hat)),
        "f1": float(f1_score(y_true, y_hat)),
        "mcc": float(matthews_corrcoef(y_true, y_hat)),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)
    return metrics


def compute_lift_table(
    y_true,
    y_proba,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return a decile lift table useful for churn targeting.

    The output contains:
    - bin
    - customers_in_bin
    - churners_in_bin
    - cumulative_customers
    - cumulative_churners
    - response_rate
    - cumulative_response_rate
    - lift
    - cumulative_lift
    """
    df = pd.DataFrame({"y_true": y_true, "y_proba": y_proba}).copy()
    df = df.sort_values("y_proba", ascending=False).reset_index(drop=True)
    # Bin by rank
    df["bin"] = pd.qcut(df.index, q=n_bins, labels=False) + 1

    total_churners = df["y_true"].sum()
    avg_rate = total_churners / len(df) if len(df) > 0 else 0.0

    agg = (
        df.groupby("bin")
        .agg(
            customers_in_bin=("y_true", "size"),
            churners_in_bin=("y_true", "sum"),
            max_score=("y_proba", "max"),
        )
        .reset_index()
        .sort_values("bin")
    )
    agg["cumulative_customers"] = agg["customers_in_bin"].cumsum()
    agg["cumulative_churners"] = agg["churners_in_bin"].cumsum()
    agg["response_rate"] = agg["churners_in_bin"] / agg["customers_in_bin"].replace(0, np.nan)
    agg["cumulative_response_rate"] = agg["cumulative_churners"] / agg["cumulative_customers"].replace(0, np.nan)

    if avg_rate > 0:
        agg["lift"] = agg["response_rate"] / avg_rate
        agg["cumulative_lift"] = agg["cumulative_response_rate"] / avg_rate
    else:
        agg["lift"] = np.nan
        agg["cumulative_lift"] = np.nan
    return agg


def compute_business_metrics(
    y_true,
    y_proba,
    config,
    n_bins: int = 10,
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Compute a few business-oriented metrics for churn campaigns.

    Includes:
    - top_10pct_capture
    - top_20pct_capture
    - optional profit-optimised threshold if cost inputs are provided.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    lift_table = compute_lift_table(y_true, y_proba, n_bins=n_bins)
    total_churners = float(y_true.sum()) if len(y_true) > 0 else 0.0

    metrics: Dict[str, float] = {}

    if total_churners > 0 and not lift_table.empty:
        top_decile_churn = float(lift_table.loc[lift_table["bin"] == 1, "churners_in_bin"].iloc[0])
        top_2dec_churn = float(lift_table.loc[lift_table["bin"] <= 2, "churners_in_bin"].sum())
        metrics["top_10pct_capture"] = top_decile_churn / total_churners
        metrics["top_20pct_capture"] = top_2dec_churn / total_churners

    # Optional profit-based threshold search
    if (
        config.churn_cost_per_customer is not None
        and config.retention_offer_cost is not None
        and config.expected_retained_value is not None
    ):
        thresholds = np.linspace(0.0, 1.0, 101)
        best_thr, best_profit = 0.5, -np.inf
        for thr in thresholds:
            y_hat = (y_proba >= thr).astype(int)
            tp = ((y_hat == 1) & (y_true == 1)).sum()
            fp = ((y_hat == 1) & (y_true == 0)).sum()

            benefit = tp * (config.expected_retained_value - config.retention_offer_cost)
            cost = fp * config.retention_offer_cost
            profit = benefit - cost
            if profit > best_profit:
                best_profit = float(profit)
                best_thr = float(thr)

        metrics["optimal_threshold"] = best_thr
        metrics["optimal_expected_profit"] = best_profit

    return metrics
