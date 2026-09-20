"""Machine learning package for educational market-direction prediction.

Provides a full pipeline: feature engineering, target construction, time-aware
training, evaluation, prediction, and explainability. The pipeline predicts
next-period market direction (up/down) as a binary classification problem with
explicit calibration, leakage guards, and educational disclaimers.
"""

from __future__ import annotations

from src.ml import evaluation, explainability, prediction, preprocessing, training

__all__ = [
    "evaluation",
    "explainability",
    "prediction",
    "preprocessing",
    "training",
]
