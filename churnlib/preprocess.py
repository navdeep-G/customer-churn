from typing import List
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_features),
            ("cat", categorical_pipe, cat_features),
        ]
    )
    return preprocessor
