"""Numeric helper utilities."""

from __future__ import annotations

import math


def is_nan(value: float | None) -> bool:
    """Return ``True`` if ``value`` is ``None`` or NaN.

    Uses :func:`math.isnan` so ruff's ``PLR0124`` "compared with itself"
    lint rule is satisfied.
    """

    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return True
