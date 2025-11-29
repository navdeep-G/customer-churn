import pandas as pd

from churnlib.config import ChurnConfig


def test_infer_features_and_validate():
    df = pd.DataFrame(
        {
            "customer_id": [1, 2, 3],
            "churn": [0, 1, 0],
            "tenure_months": [10, 3, 24],
            "plan_type": ["A", "B", "A"],
        }
    )
    cfg = ChurnConfig(
        id_col="customer_id",
        label_col="churn",
    )
    cfg.validate(df)
    cfg.infer_features(df)

    assert "tenure_months" in cfg.num_features
    assert "plan_type" in cfg.cat_features
