"""Frame validation utilities for the market data layer.

These helpers centralise the rules for deciding whether a response
returned by a provider is "good enough" to be used by the dashboard:

* Required OHLCV columns must be present.
* A minimum number of rows must be available.
* Numeric values must be finite.
* All-NaN rows are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.market_data_service.errors import MarketDataError, SymbolNotFoundError
from src.utils.logging import get_logger
from src.utils.numeric import is_nan

logger = get_logger(__name__)

REQUIRED_OHLCV_COLUMNS: tuple[str, ...] = ("Open", "High", "Low", "Close", "Volume")


@dataclass(frozen=True)
class FrameValidationResult:
    """Outcome of a frame validation."""

    is_valid: bool
    row_count: int
    reason: str = ""


def _ensure_columns(frame: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        raise MarketDataError(f"OHLCV frame is missing required columns: {missing}")


def normalise_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    drop_all_nan_rows: bool = True,
) -> pd.DataFrame:
    """Return a canonical OHLCV frame.

    The function guarantees:

    * A :class:`pandas.DatetimeIndex` (UTC when the provider returns a
      naive index).
    * The canonical column order ``Open, High, Low, Close, Volume``.
    * Numeric dtypes for the price/volume columns.
    * Rows that are all-NaN removed when ``drop_all_nan_rows`` is
      ``True``.

    Missing columns are filled with :data:`pandas.NA` for price columns
    and zero for ``Volume`` so downstream validation can decide what to
    do with partial frames. Use :func:`validate_ohlcv_frame` or
    :func:`require_valid_frame` to assert correctness.
    """

    if frame is None:
        return pd.DataFrame(columns=list(REQUIRED_OHLCV_COLUMNS))
    if frame.empty:
        return frame.copy()

    working = frame.copy()
    for required in REQUIRED_OHLCV_COLUMNS:
        if required not in working.columns:
            working[required] = pd.NA
    working = working[list(REQUIRED_OHLCV_COLUMNS)]

    if not isinstance(working.index, pd.DatetimeIndex):
        try:
            working.index = pd.to_datetime(working.index)
        except (TypeError, ValueError) as exc:
            raise MarketDataError("OHLCV frame index is not coercible to datetime") from exc

    for column in ("Open", "High", "Low", "Close"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["Volume"] = pd.to_numeric(working["Volume"], errors="coerce").fillna(0).astype("int64")

    if drop_all_nan_rows:
        price_cols = ["Open", "High", "Low", "Close"]
        all_nan_mask = working[price_cols].isna().all(axis=1)
        if all_nan_mask.any():
            dropped = int(all_nan_mask.sum())
            logger.debug("Dropping %d all-NaN OHLCV rows", dropped)
            working = working.loc[~all_nan_mask]

    return working.sort_index()


def _basic_shape_check(
    frame: pd.DataFrame | None, min_rows: int
) -> FrameValidationResult:
    invalid: FrameValidationResult | None = None
    if frame is None or frame.empty:
        invalid = FrameValidationResult(is_valid=False, row_count=0, reason="empty frame")
    elif not isinstance(frame, pd.DataFrame):
        invalid = FrameValidationResult(is_valid=False, row_count=0, reason="not a DataFrame")
    else:
        try:
            _ensure_columns(frame)
        except MarketDataError as exc:
            invalid = FrameValidationResult(
                is_valid=False, row_count=int(len(frame)), reason=str(exc)
            )
        else:
            row_count = int(len(frame))
            if row_count < min_rows:
                invalid = FrameValidationResult(
                    is_valid=False,
                    row_count=row_count,
                    reason=f"insufficient rows ({row_count} < {min_rows})",
                )
            else:
                closes = frame["Close"]
                if closes.isna().all():
                    invalid = FrameValidationResult(
                        is_valid=False,
                        row_count=row_count,
                        reason="all-NaN Close column",
                    )
                elif not np.isfinite(closes.dropna().to_numpy()).all():
                    invalid = FrameValidationResult(
                        is_valid=False,
                        row_count=row_count,
                        reason="non-finite Close values",
                    )
    if invalid is not None:
        return invalid
    return FrameValidationResult(is_valid=True, row_count=int(len(frame)))


def validate_ohlcv_frame(
    frame: pd.DataFrame,
    *,
    min_rows: int = 1,
) -> FrameValidationResult:
    """Return a :class:`FrameValidationResult` for ``frame``.

    The result is informational; it never raises. The function is
    cheap to call repeatedly and is used to gate downstream
    computations (for example RSI which requires at least 14 rows).
    """

    return _basic_shape_check(frame, min_rows)


def require_valid_frame(
    frame: pd.DataFrame,
    *,
    symbol: str,
    min_rows: int = 1,
) -> pd.DataFrame:
    """Validate ``frame`` and raise if it is unusable.

    An empty frame is mapped to :class:`SymbolNotFoundError` because
    that is the most common upstream signal for an invalid ticker. A
    malformed frame raises :class:`MarketDataError`.
    """

    if frame is None or frame.empty:
        raise SymbolNotFoundError(f"No data available for symbol '{symbol}'")

    result = validate_ohlcv_frame(frame, min_rows=min_rows)
    if not result.is_valid:
        if "missing required columns" in result.reason or "non-finite" in result.reason:
            raise MarketDataError(f"Malformed OHLCV data for '{symbol}': {result.reason}")
        raise SymbolNotFoundError(
            f"Insufficient usable data for '{symbol}': {result.reason}"
        )
    return frame


def safe_last_close(frame: pd.DataFrame) -> float | None:
    """Return the last non-NaN close as ``float`` or ``None``."""

    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    series = frame["Close"].dropna()
    if series.empty:
        return None
    value = float(series.iloc[-1])
    return None if is_nan(value) else value


def safe_previous_close(frame: pd.DataFrame) -> float | None:
    """Return the second-to-last non-NaN close or ``None``."""

    if frame is None or frame.empty or "Close" not in frame.columns:
        return None
    series = frame["Close"].dropna()
    if len(series) < 2:
        return None
    value = float(series.iloc[-2])
    return None if is_nan(value) else value
