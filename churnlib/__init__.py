"""churnlib: a lightweight toolkit for customer churn modelling.

The main entry point for most users is :class:`ChurnProject`.
"""

from .config import ChurnConfig
from .experiment import ChurnProject, ChurnResults

__all__ = ["ChurnConfig", "ChurnProject", "ChurnResults"]
