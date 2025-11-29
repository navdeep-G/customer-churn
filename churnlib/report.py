from typing import List

import html
from .experiment import ChurnResults


def _dict_to_html_table(d: dict) -> str:
    rows = []
    for k, v in d.items():
        rows.append(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>")
    return "<table>" + "".join(rows) + "</table>"


def render_html_report(results: ChurnResults) -> str:
    """Render a minimal self-contained HTML report.

    The report is intentionally lightweight so that it can be versioned
    or attached to experiment artefacts. For richer reports, users can
    build on the underlying metrics and dataframes.
    """
    cfg = results.config

    title = "Churn modelling report"

    header = f"<h1>{html.escape(title)}</h1>"

    config_bits = {
        "ID column": cfg.id_col,
        "Label column": cfg.label_col,
        "Positive label": cfg.positive_label,
        "Date column": cfg.date_col or "(none)",
        "Prediction horizon (days)": cfg.prediction_horizon_days or "(unspecified)",
        "Numeric features": ", ".join(cfg.num_features or []),
        "Categorical features": ", ".join(cfg.cat_features or []),
    }
    config_html = "<h2>Configuration</h2>" + _dict_to_html_table(config_bits)

    metrics_html = "<h2>Model performance (test set)</h2>" + _dict_to_html_table(results.metrics)

    business_html = "<h2>Business metrics</h2>" + _dict_to_html_table(results.business_metrics)

    fi_head = results.feature_importance.head(20).to_html(index=False, escape=True)
    fi_html = "<h2>Top features</h2>" + fi_head

    lift_head = results.lift_table.head(10).to_html(index=False, escape=True)
    lift_html = "<h2>Lift table (top bins)</h2>" + lift_head

    body = header + config_html + metrics_html + business_html + fi_html + lift_html

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{html.escape(title)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 2rem;
      line-height: 1.5;
    }}
    table {{
      border-collapse: collapse;
      margin-bottom: 1.5rem;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 0.4rem 0.6rem;
    }}
    th {{
      text-align: left;
      background: #f7f7f7;
    }}
    h1, h2 {{
      margin-top: 1.2rem;
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
    return html_doc
