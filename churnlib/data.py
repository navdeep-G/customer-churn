from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_train_test_split(
    df: pd.DataFrame,
    label_col: str,
    date_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a train/test split for churn modelling.

    If ``date_col`` is provided, the split is time-based: the most recent
    observations (by date) form the test set. Otherwise a stratified random
    split is used.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    if date_col and date_col in df.columns:
        df_sorted = df.sort_values(date_col)
        n_total = len(df_sorted)
        n_test = int(round(test_size * n_total))
        if n_test <= 0 or n_test >= n_total:
            raise ValueError("test_size results in empty train or test set.")
        test = df_sorted.iloc[-n_test:]
        train = df_sorted.iloc[:-n_test]
        X_train = train.drop(columns=[label_col])
        X_test = test.drop(columns=[label_col])
        y_train = train[label_col]
        y_test = test[label_col]
        return X_train, X_test, y_train, y_test

    # Stratified random split on label
    X = df.drop(columns=[label_col])
    y = df[label_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    return X_train, X_test, y_train, y_test
