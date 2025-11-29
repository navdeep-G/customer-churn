from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pandas as pd


@dataclass
class ChurnModelSelector:
    """Simple model selector for churn modelling.

    Evaluates a small set of baseline models using ROC AUC and returns
    the best-performing pipeline.
    """

    cv_folds: int = 5
    random_state: int = 42

    def _candidate_models(self) -> List[Tuple[str, object]]:
        return [
            ("logistic_regression", LogisticRegression(max_iter=1000)),
            ("gradient_boosting", GradientBoostingClassifier(random_state=self.random_state)),
        ]

    def fit_best(
        self,
        preprocessor,
        X_train,
        y_train,
    ) -> Tuple[Pipeline, str]:
        """Fit candidate models and return the best pipeline and its name."""
        X_train = X_train.copy()
        y_train = np.asarray(y_train)

        cv = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        best_name = None
        best_score = -np.inf
        best_pipeline = None

        for name, estimator in self._candidate_models():
            pipe = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("model", estimator),
                ]
            )
            scores = cross_val_score(
                pipe,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
            )
            mean_score = float(scores.mean())
            if mean_score > best_score:
                best_score = mean_score
                best_name = name
                best_pipeline = pipe

        if best_pipeline is None:
            raise RuntimeError("No models were successfully fitted.")

        best_pipeline.fit(X_train, y_train)
        return best_pipeline, best_name

    def compute_feature_importance(
        self,
        pipeline: Pipeline,
    ) -> pd.DataFrame:
        """Compute a simple global feature importance table.

        For tree-based models this uses ``feature_importances_``.
        For linear models it uses absolute coefficients.
        For other models, an empty DataFrame is returned.
        """
        pre = pipeline.named_steps.get("preprocess")
        model = pipeline.named_steps.get("model")
        try:
            feature_names = pre.get_feature_names_out()
        except Exception:
            feature_names = [f"f_{i}" for i in range(len(getattr(model, "feature_importances_", [])))]

        importances = None
        if hasattr(model, "feature_importances_"):
            importances = getattr(model, "feature_importances_")
        elif hasattr(model, "coef_"):
            coef = getattr(model, "coef_")
            if coef.ndim == 2:
                coef = coef[0]
            importances = abs(coef)

        if importances is None:
            return pd.DataFrame(columns=["feature", "importance"])

        if len(feature_names) != len(importances):
            # Fallback in odd edge-cases
            feature_names = [f"f_{i}" for i in range(len(importances))]

        df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        return df.reset_index(drop=True)
