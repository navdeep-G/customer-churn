import numpy as np
import pandas as pd

from churnlib import ChurnProject


def make_synthetic_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    customer_id = np.arange(n)
    tenure = rng.integers(1, 36, size=n)
    monthly_spend = rng.normal(50, 10, size=n).clip(5, 200)
    plan_type = rng.choice(["A", "B", "C"], size=n)

    # Simple churn mechanism: short tenure & high spend => more likely to churn
    logits = -2.0 + 0.05 * (200 - monthly_spend) - 0.1 * tenure
    probs = 1 / (1 + np.exp(-logits))
    churn = (rng.random(size=n) < probs).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "churn": churn,
            "tenure": tenure,
            "monthly_spend": monthly_spend,
            "plan_type": plan_type,
        }
    )
    return df


def test_churn_project_end_to_end():
    df = make_synthetic_df()

    project = ChurnProject.from_dataframe(
        df,
        id_col="customer_id",
        label_col="churn",
        positive_label=1,
    )

    results = project.auto_fit(df)
    summary = project.summary()

    assert "best_model" in summary
    assert "metrics" in summary
    assert "roc_auc" in summary["metrics"]
    assert results.lift_table is not None
    assert not results.feature_importance.empty

    scores = project.score(df.head(10))
    assert "churn_probability" in scores.columns
    assert len(scores) == 10
