from dataclasses import dataclass, field
from typing import List, Optional, Hashable

import pandas as pd


@dataclass
class ChurnConfig:
    """Configuration for a churn modelling project.

    Parameters
    ----------
    id_col:
        Name of the unique customer identifier column.
    label_col:
        Name of the binary churn label column.
    positive_label:
        Value in ``label_col`` that represents churn (e.g. 1 or "Yes").
    date_col:
        Optional column with a snapshot / observation date. When provided,
        time-based splitting can be used to avoid leakage.
    prediction_horizon_days:
        Optional metadata to record the prediction horizon for documentation.
    num_features:
        Optional list of numeric feature column names. If omitted, inferred.
    cat_features:
        Optional list of categorical feature column names. If omitted, inferred.
    drop_cols:
        Columns to drop from modelling (IDs, free text, etc).
    churn_cost_per_customer, retention_offer_cost, expected_retained_value:
        Optional inputs used to compute simple profit-based business metrics.
    """

    id_col: str
    label_col: str
    positive_label: Hashable = 1
    date_col: Optional[str] = None
    prediction_horizon_days: Optional[int] = None

    num_features: Optional[List[str]] = None
    cat_features: Optional[List[str]] = None
    drop_cols: List[str] = field(default_factory=list)

    churn_cost_per_customer: Optional[float] = None
    retention_offer_cost: Optional[float] = None
    expected_retained_value: Optional[float] = None

    def infer_features(self, df: pd.DataFrame) -> None:
        """Infer numeric and categorical features from a dataframe.

        Columns listed in ``id_col``, ``label_col``, ``date_col`` and
        ``drop_cols`` are excluded from inference.
        """
        reserved = {self.id_col, self.label_col}
        if self.date_col:
            reserved.add(self.date_col)
        reserved.update(self.drop_cols)

        candidate_cols = [c for c in df.columns if c not in reserved]

        inferred_num, inferred_cat = [], []
        for c in candidate_cols:
            s = df[c]
            if pd.api.types.is_numeric_dtype(s):
                inferred_num.append(c)
            else:
                inferred_cat.append(c)

        if self.num_features is None:
            self.num_features = inferred_num
        if self.cat_features is None:
            self.cat_features = inferred_cat

    def validate(self, df: pd.DataFrame) -> None:
        """Basic validation of configuration against a dataframe."""
        missing = []
        for c in (self.id_col, self.label_col, self.date_col):
            if c and c not in df.columns:
                missing.append(c)
        if missing:
            raise ValueError(f"Missing required columns in dataframe: {missing}")
        if self.id_col == self.label_col:
            raise ValueError("id_col and label_col must be different.")
        if self.num_features is not None:
            missing_num = [c for c in self.num_features if c not in df.columns]
            if missing_num:
                raise ValueError(f"Unknown numeric features: {missing_num}")
        if self.cat_features is not None:
            missing_cat = [c for c in self.cat_features if c not in df.columns]
            if missing_cat:
                raise ValueError(f"Unknown categorical features: {missing_cat}")
