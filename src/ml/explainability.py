"""Conservative model explainability helpers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def feature_importance(
    feature_names: Iterable[str],
    importances: Iterable[float],
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Rank feature scores by absolute magnitude without causal interpretation."""

    names = list(feature_names)
    values = [float(value) for value in importances]
    if len(names) != len(values):
        raise ValueError("feature_names and importances must have the same length")
    pairs = sorted(
        ({"feature": name, "importance": value} for name, value in zip(names, values, strict=True)),
        key=lambda item: abs(item["importance"]),
        reverse=True,
    )
    return pairs if top_n is None else pairs[:top_n]


def explain_prediction(
    estimator: Any,
    feature_names: Iterable[str],
    row: pd.Series | pd.DataFrame,
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Return local linear contributions or global tree scores.

    Tree impurity importances are not local explanations.  The returned
    ``scope`` field makes that distinction explicit.
    """

    names = list(feature_names)
    values = np.asarray(
        row.iloc[0].to_numpy() if isinstance(row, pd.DataFrame) else row.to_numpy(),
        dtype=float,
    )
    if len(names) != len(values):
        raise ValueError("feature_names and row must have the same length")
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        if coefficients.size == 1 and len(names) > 1:
            coefficients = np.repeat(coefficients[0], len(names))
        contributions = coefficients[: len(names)] * values
        records = [
            {
                "feature": name,
                "contribution": float(contribution),
                "absolute_contribution": float(abs(contribution)),
                "scope": "local_linear",
                "interpretation": "association, not causation",
            }
            for name, contribution in zip(names, contributions, strict=True)
        ]
    elif hasattr(estimator, "feature_importances_"):
        scores = np.asarray(estimator.feature_importances_, dtype=float)
        records = [
            {
                "feature": name,
                "importance": float(score),
                "absolute_contribution": float(abs(score)),
                "scope": "global_tree",
                "interpretation": "global split importance, not a local contribution",
            }
            for name, score in zip(names, scores, strict=True)
        ]
        records.sort(key=lambda item: item["absolute_contribution"], reverse=True)
    else:
        return []
    records.sort(key=lambda item: item["absolute_contribution"], reverse=True)
    return records[:top_n]


def explain_model(
    estimator: Any,
    feature_names: Iterable[str],
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Explain a fitted estimator using the safest available model attribute."""

    names = list(feature_names)
    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float).reshape(-1)
        if coefficients.size == 1 and len(names) > 1:
            coefficients = np.repeat(coefficients[0], len(names))
        return feature_importance(names, coefficients[: len(names)], top_n=top_n)
    if hasattr(estimator, "feature_importances_"):
        return feature_importance(
            names, np.asarray(estimator.feature_importances_, dtype=float)[: len(names)], top_n=top_n
        )
    return []
