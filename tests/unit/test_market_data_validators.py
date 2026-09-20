"""Tests for the OHLCV validation utilities."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data.market_data_service.errors import MarketDataError, SymbolNotFoundError
from src.data.market_data_service.validators import (
    normalise_ohlcv_frame,
    require_valid_frame,
    safe_last_close,
    safe_previous_close,
    validate_ohlcv_frame,
)


def _frame(rows: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, rows),
            "High": np.linspace(101, 111, rows),
            "Low": np.linspace(99, 109, rows),
            "Close": np.linspace(100, 110, rows),
            "Volume": [1_000_000] * rows,
        },
        index=idx,
    )


def test_normalise_returns_canonical_columns_and_dtypes() -> None:
    raw = _frame(10)
    # Rename Close to mixed case to ensure normalisation handles casing.
    raw = raw.rename(columns={"close": "CLOSE"} if "CLOSE" in raw.columns else {})
    out = normalise_ohlcv_frame(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert out["Volume"].dtype.kind in ("i", "u")
    assert isinstance(out.index, pd.DatetimeIndex)


def test_normalise_drops_all_nan_rows() -> None:
    raw = _frame(5)
    raw.loc[raw.index[2], ["Open", "High", "Low", "Close"]] = np.nan
    out = normalise_ohlcv_frame(raw)
    assert len(out) == 4


def test_normalise_handles_empty_input() -> None:
    assert normalise_ohlcv_frame(pd.DataFrame()).empty
    assert normalise_ohlcv_frame(None).empty  # type: ignore[arg-type]


def test_normalise_fills_missing_columns() -> None:
    raw = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    out = normalise_ohlcv_frame(raw)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(out) == 3
    assert out["Open"].isna().all()  # filled with NA
    assert int(out["Volume"].sum()) == 0


def test_validate_ohlcv_frame_reports_insufficient_rows() -> None:
    result = validate_ohlcv_frame(_frame(2), min_rows=10)
    assert not result.is_valid
    assert "insufficient" in result.reason


def test_validate_ohlcv_frame_rejects_non_finite() -> None:
    raw = _frame(5)
    raw.loc[raw.index[0], "Close"] = math.inf
    result = validate_ohlcv_frame(raw)
    assert not result.is_valid
    assert "non-finite" in result.reason


def test_require_valid_frame_raises_on_empty() -> None:
    with pytest.raises(SymbolNotFoundError):
        require_valid_frame(pd.DataFrame(), symbol="AAPL")


def test_require_valid_frame_raises_on_missing_columns() -> None:
    raw = _frame(10).drop(columns=["Close"])
    with pytest.raises(MarketDataError):
        require_valid_frame(raw, symbol="AAPL")


def test_safe_last_close_handles_missing_and_nan() -> None:
    assert safe_last_close(pd.DataFrame()) is None
    empty = pd.DataFrame({"Close": []})
    assert safe_last_close(empty) is None
    nan_frame = pd.DataFrame({"Close": [float("nan")]})
    assert safe_last_close(nan_frame) is None
    # _frame(3) Close column: [100.0, 105.0, 110.0]
    assert safe_last_close(_frame(3)) == pytest.approx(110.0)


def test_safe_previous_close_requires_two_rows() -> None:
    assert safe_previous_close(_frame(1)) is None
    # _frame(3) Close: [100.0, 105.0, 110.0] -> previous is 105.0
    assert safe_previous_close(_frame(3)) == pytest.approx(105.0)
