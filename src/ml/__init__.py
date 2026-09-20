"""Machine learning package.

This package is intentionally lightweight in the current refactor. It
provides thin interfaces that the rest of the codebase can rely on without
tying the application to any specific model implementation. Advanced
models will be added in a follow-up.
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
