from typing import Optional

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline


def permutation_feature_importance(
    pipeline: Pipeline,
    X_valid,
    y_valid,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute permutation feature importance for a fitted pipeline.

    This is slower than model-based feature importance but works for a
    wider variety of models.
    """
    result = permutation_importance(
        pipeline,
        X_valid,
        y_valid,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1,
    )
    pre = pipeline.named_steps.get("preprocess")
    try:
        feature_names = pre.get_feature_names_out()
    except Exception:
        feature_names = [f"f_{i}" for i in range(len(result.importances_mean))]
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return df.reset_index(drop=True)
