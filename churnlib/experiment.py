from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from .config import ChurnConfig
from .data import make_train_test_split
from .preprocess import build_preprocessor
from .models import ChurnModelSelector
from .metrics import (
    compute_classification_metrics,
    compute_business_metrics,
    compute_lift_table,
)


@dataclass
class ChurnResults:
    """Container for the results of a churn modelling run."""

    config: ChurnConfig
    best_model_name: str
    metrics: Dict[str, Any]
    business_metrics: Dict[str, Any]
    lift_table: pd.DataFrame
    feature_importance: pd.DataFrame


class ChurnProject:
    """High-level orchestration class for churn modelling.

    Typical usage
    -------------
    >>> project = ChurnProject.from_dataframe(
    ...     df,
    ...     id_col="customer_id",
    ...     label_col="churn",
    ...     positive_label=1,
    ... )
    >>> project.auto_fit(df)
    >>> project.summary()
    >>> project.report("churn_report.html")
    """

    def __init__(self, config: ChurnConfig):
        self.config = config
        self._model = None
        self._results: Optional[ChurnResults] = None

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, **config_kwargs) -> "ChurnProject":
        config = ChurnConfig(**config_kwargs)
        config.validate(df)
        config.infer_features(df)
        return cls(config)

    def auto_fit(self, df: pd.DataFrame) -> ChurnResults:
        """Run the end-to-end churn workflow on a dataframe.

        This method:
        - splits the data into train/test
        - builds preprocessing
        - selects and fits a model
        - computes metrics and lift table
        """
        cfg = self.config

        # Restrict to modelling columns
        all_features = (cfg.num_features or []) + (cfg.cat_features or [])
        modelling_cols = list(dict.fromkeys(all_features + [cfg.label_col]))

        missing = [c for c in modelling_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Dataframe is missing required columns: {missing}")

        df_mod = df[modelling_cols].copy()

        X_train, X_test, y_train, y_test = make_train_test_split(
            pd.concat([df_mod.drop(columns=[cfg.label_col]), df_mod[cfg.label_col]], axis=1),
            label_col=cfg.label_col,
            date_col=cfg.date_col,
        )

        preprocessor = build_preprocessor(
            num_features=cfg.num_features or [],
            cat_features=cfg.cat_features or [],
        )

        selector = ChurnModelSelector()
        model, model_name = selector.fit_best(
            preprocessor,
            X_train,
            y_train,
        )

        self._model = model

        y_test_array = np.asarray(y_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = compute_classification_metrics(
            y_true=y_test_array,
            y_proba=y_proba,
            positive_label=cfg.positive_label,
        )
        business_metrics = compute_business_metrics(
            y_true=y_test_array,
            y_proba=y_proba,
            config=cfg,
        )
        lift_table = compute_lift_table(y_test_array, y_proba)

        feature_importance = selector.compute_feature_importance(model)

        self._results = ChurnResults(
            config=cfg,
            best_model_name=model_name,
            metrics=metrics,
            business_metrics=business_metrics,
            lift_table=lift_table,
            feature_importance=feature_importance,
        )
        return self._results

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serialisable summary of the run."""
        if self._results is None:
            raise RuntimeError("Call auto_fit() first.")
        return {
            "best_model": self._results.best_model_name,
            "metrics": self._results.metrics,
            "business_metrics": self._results.business_metrics,
        }

    def report(self, path: str) -> None:
        """Generate a simple HTML report summarising the project."""
        from .report import render_html_report

        if self._results is None:
            raise RuntimeError("Call auto_fit() first.")
        html = render_html_report(self._results)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score new customers and return churn probabilities.

        The input dataframe must contain the numeric and categorical
        feature columns defined in the configuration, plus the ID column.
        """
        if self._model is None or self._results is None:
            raise RuntimeError("Call auto_fit() first.")

        cfg = self.config
        all_features = (cfg.num_features or []) + (cfg.cat_features or [])
        missing = [c for c in all_features + [cfg.id_col] if c not in df.columns]
        if missing:
            raise ValueError(f"Dataframe is missing required columns: {missing}")

        X = df[all_features].copy()
        proba = self._model.predict_proba(X)[:, 1]

        out = df[[cfg.id_col]].copy()
        out["churn_probability"] = proba
        return out
