# churnlib

`churnlib` is a small Python toolkit that helps data scientists run
end‑to‑end customer churn analyses with just a few lines of code.

## Why churn analysis matters

Customer churn is one of the most direct levers for sustainable growth: it impacts revenue predictability, customer lifetime value (CLV), and how efficiently you can spend on acquisition and retention.

A few practical reasons teams invest in churn analysis:

- **Retention is usually cheaper than acquisition.** HBR summarizes research suggesting acquiring a new customer can be *5–25×* more expensive than retaining an existing one. :contentReference[oaicite:1]{index=1}
- **Small retention gains can have outsized profit impact.** HBR also cites work (Reichheld / Bain) often quoted as: improving retention by **5%** can increase profits by **25%–95%** (industry-dependent). :contentReference[oaicite:2]{index=2}
- **Operational focus:** churn analysis helps you identify *who is at risk*, *why*, and *what intervention is worth it* (discount vs. onboarding help vs. product fix).
- **Better targeting:** instead of blanket outreach, you can prioritize customers where preventing churn produces the highest incremental value (often via lift / gain style views).

### What “good” churn analysis includes

Beyond a single churn rate, teams typically want:

- A clear **label definition** (what counts as churn, and over what horizon?)
- A **time-aware split** (to avoid leakage when customer behavior changes over time)
- ML metrics **and** business-facing views (e.g., **lift tables** / “top decile captures X% of churners”)
- **Interpretability** (which features drive risk) so the analysis leads to action

`churnlib` is designed to cover these basics end-to-end: preprocessing, baseline modeling, standard ML metrics, churn-specific business metrics + lift tables, and a lightweight HTML report you can share with stakeholders.

## Features

* Simple configuration via `ChurnConfig`
* Automatic train/test split (time‑based or random)
* Sensible preprocessing for numeric and categorical features
* Baseline model selection (logistic regression & gradient boosting)
* Standard ML metrics (AUC, PR‑AUC, F1, etc.)
* Churn‑specific business metrics and lift table
* Simple HTML report summarising configuration, performance, and top features

## Installation (local / editable)

From the directory that contains `pyproject.toml`:

```bash
pip install -e .
```

This will install `churnlib` in editable mode so changes to the source
are picked up immediately.

## Quickstart

```python
import pandas as pd
from churnlib import ChurnProject

df = pd.read_csv("your_customer_table.csv")

project = ChurnProject.from_dataframe(
    df,
    id_col="customer_id",
    label_col="churn",
    positive_label=1,   # or "Yes"
    date_col="snapshot_date",        # optional
    prediction_horizon_days=30,      # optional, for documentation
)

# Run the full workflow: split, preprocess, model selection, evaluation
results = project.auto_fit(df)

# Notebook‑friendly summary
print(project.summary())

# Generate a simple HTML report
project.report("churn_report.html")

# Score new customers
scoring_df = df.sample(100).copy()
scores = project.score(scoring_df)
print(scores.head())
```

## Dependencies

Key dependencies (also listed in `pyproject.toml`):

* pandas
* scikit‑learn
* matplotlib (for optional plotting helpers)

## Running tests

```bash
pip install -e ".[dev]"
pytest
```
