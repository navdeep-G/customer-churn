from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import inspect


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a OneHotEncoder that works across sklearn versions.

    - For newer versions (with 'sparse_output'): use sparse_output=False
    - For older versions (with 'sparse'): use sparse=False
    """
    params = inspect.signature(OneHotEncoder).parameters
    if "sparse_output" in params:
        # sklearn >= 1.4
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    elif "sparse" in params:
        # older sklearn (<= 1.3)
        return OneHotEncoder(handle_unknown="ignore", sparse=False)
    else:
        # Fallback: don't specify sparsity arg
        return OneHotEncoder(handle_unknown="ignore")


def build_preprocessor(
    num_features: List[str],
    cat_features: List[str],
) -> ColumnTransformer:
    """Build a preprocessing transformer for churn modelling.

    - Numeric features: median imputation + standard scaling.
    - Categorical features: most-frequent imputation + one-hot encoding.
    """
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", _make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_features),
            ("cat", categorical_pipe, cat_features),
        ]
    )
    return preprocessor

