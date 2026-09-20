"""Leakage-safe feature engineering and time-series preprocessing.

The target is deliberately constructed separately from the feature frame.  A
feature row at time ``t`` may use observations at or before ``t`` only; the
target uses the close at ``t + horizon`` and is excluded when that future
observation is unavailable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.features.analytics import (
    adx,
    atr,
    bollinger_bands,
    drawdown,
    macd,
    obv,
    rolling_volatility,
    rsi,
)
from src.features.technical_indicators import calculate_moving_averages

TARGET_DEFINITION = (
    "Binary direction of the simple Close return over the next horizon: "
    "1 when the return is above the configured positive threshold and 0 when "
    "it is below the negative threshold. The default threshold is 0.0; a "
    "positive threshold creates a neutral band whose rows are NaN, as are rows "
    "without an available future close."
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "price": (
        "return_1",
        "log_return_1",
        "rolling_return_5",
        "rolling_return_20",
        "rolling_return_50",
        "volatility_20",
        "drawdown",
    ),
    "technical": (
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_histogram",
        "MA_5",
        "MA_20",
        "MA_50",
        "price_ma_ratio_5",
        "price_ma_ratio_20",
        "price_ma_ratio_50",
        "bollinger_position",
        "bollinger_width",
        "atr_percent",
        "adx",
        "obv",
        "volume_change",
        "volume_zscore_20",
    ),
}


def _feature_groups(feature_names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Return static feature groups plus any aligned market-context features."""

    names = list(feature_names)
    groups = {group: tuple(name for name in values if name in names) for group, values in FEATURE_GROUPS.items()}
    context_suffixes = ("_return_1", "_volatility_20", "_volatility_indicator")
    context_names = tuple(
        name
        for name in names
        if any(name.endswith(suffix) for suffix in context_suffixes)
    )
    if context_names:
        groups["market_context"] = context_names
    return groups


def _validate_frame(data: pd.DataFrame, *, require_ohlcv: bool = False) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("data must have a DatetimeIndex")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data must be sorted by its DatetimeIndex")
    if data.index.has_duplicates:
        raise ValueError("data index must be unique")

    frame = data.copy()
    required = ["Close"]
    if require_ohlcv:
        required.extend(["Open", "High", "Low", "Volume"])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    for column in required:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_return(close: pd.Series, window: int) -> pd.Series:
    return close.pct_change(periods=window, fill_method=None)


def _context_feature_frame(
    context: pd.DataFrame,
    prefix: str,
    target_index: pd.DatetimeIndex,
    lag: int,
) -> pd.DataFrame:
    context = _validate_frame(context)
    if "Close" not in context.columns:
        raise KeyError("context frames must contain Close")
    close = pd.to_numeric(context["Close"], errors="coerce")
    log_return = np.log(close / close.shift(1)).where(close > 0)
    result = pd.DataFrame(
        {
            f"{prefix}_return_1": close.pct_change(fill_method=None),
            f"{prefix}_volatility_20": log_return.rolling(20, min_periods=20).std(ddof=1),
        },
        index=context.index,
    )
    if "Volatility" in context.columns or "volatility" in context.columns:
        column = "Volatility" if "Volatility" in context.columns else "volatility"
        result[f"{prefix}_volatility_indicator"] = pd.to_numeric(
            context[column], errors="coerce"
        )
    return result.shift(lag).reindex(target_index)


def build_feature_frame(
    data: pd.DataFrame,
    price_column: str = "Close",
    windows: Iterable[int] = (5, 20, 50),
    context_frames: Mapping[str, pd.DataFrame] | None = None,
    context_lag: int = 1,
) -> pd.DataFrame:
    """Build only causal price, technical, volume, and optional context features.

    Raw OHLCV columns are intentionally omitted.  Context frames are computed
    independently and shifted by ``context_lag`` before alignment, so a context
    value observed at time ``t`` cannot enter the feature row at ``t``.
    """

    if data is None or data.empty:
        return pd.DataFrame()
    frame = _validate_frame(data)
    if price_column not in frame.columns:
        raise KeyError(f"Column '{price_column}' not present in data")
    close = pd.to_numeric(frame[price_column], errors="coerce")
    if close.isna().any():
        raise ValueError("price_column contains missing or non-numeric values")

    clean_windows = sorted({int(window) for window in windows})
    if not clean_windows or clean_windows[0] < 1:
        raise ValueError("windows must contain positive integers")
    if not isinstance(context_lag, int) or context_lag < 1:
        raise ValueError("context_lag must be a positive integer")

    log_return = np.log(close / close.shift(1)).where(close > 0)
    result = pd.DataFrame(index=frame.index)
    result["return_1"] = close.pct_change(fill_method=None)
    result["log_return_1"] = log_return
    for window in clean_windows:
        result[f"rolling_return_{window}"] = _rolling_return(close, window)
    result["volatility_20"] = rolling_volatility(
        frame, window=20, price_column=price_column, annualize=False
    )
    result["drawdown"] = drawdown(frame, price_column=price_column)

    result["rsi_14"] = rsi(frame, window=14, price_column=price_column)
    macd_values = macd(frame, price_column=price_column)
    result["macd"] = macd_values["macd"]
    result["macd_signal"] = macd_values["signal"]
    result["macd_histogram"] = macd_values["histogram"]

    moving_averages = calculate_moving_averages(
        frame, windows=clean_windows, price_column=price_column
    )
    for name, values in moving_averages.items():
        result[name] = values
    for window in clean_windows:
        ma_column = f"MA_{window}"
        result[f"price_ma_ratio_{window}"] = _safe_ratio(close, result[ma_column])

    if {"High", "Low"}.issubset(frame.columns):
        bollinger = bollinger_bands(frame, window=20, price_column=price_column)
        result["bollinger_position"] = _safe_ratio(
            close - bollinger["lower"], bollinger["upper"] - bollinger["lower"]
        )
        result["bollinger_width"] = _safe_ratio(
            bollinger["upper"] - bollinger["lower"], bollinger["middle"]
        )
        result["atr_percent"] = atr(frame, window=14) / close * 100
        result["adx"] = adx(frame, window=14)
    else:
        for column in ("bollinger_position", "bollinger_width", "atr_percent", "adx"):
            result[column] = np.nan

    if "Volume" in frame.columns:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        result["obv"] = obv(frame) if "Close" in frame.columns else np.nan
        result["volume_change"] = volume.pct_change(fill_method=None)
        volume_mean = volume.rolling(20, min_periods=20).mean()
        volume_std = volume.rolling(20, min_periods=20).std(ddof=1)
        result["volume_zscore_20"] = _safe_ratio(volume - volume_mean, volume_std)
    else:
        for column in ("obv", "volume_change", "volume_zscore_20"):
            result[column] = np.nan

    for name, context in (context_frames or {}).items():
        safe_prefix = "".join(character if character.isalnum() else "_" for character in name)
        result = pd.concat(
            [
                result,
                _context_feature_frame(context, safe_prefix, frame.index, context_lag),
            ],
            axis=1,
        )

    return result.replace([np.inf, -np.inf], np.nan)


def build_target(
    data: pd.DataFrame,
    horizon: int = 1,
    price_column: str = "Close",
    threshold: float = 0.0,
) -> pd.Series:
    """Construct the next-period direction target without filling neutral rows."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    frame = _validate_frame(data)
    close = pd.to_numeric(frame[price_column], errors="coerce")
    forward_return = _safe_ratio(close.shift(-horizon), close) - 1
    target = pd.Series(np.nan, index=frame.index, dtype=float, name=f"target_{horizon}d")
    target[forward_return > threshold] = 1
    target[forward_return < -threshold] = 0
    target.attrs["definition"] = TARGET_DEFINITION
    target.attrs["horizon"] = horizon
    return target


create_target = build_target
build_classification_target = build_target


def align_features_target(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inner-align features and target, retaining feature NaNs for imputation."""

    if features.empty or target.empty:
        return features.iloc[0:0], target.iloc[0:0]
    if not isinstance(features.index, pd.DatetimeIndex) or not isinstance(
        target.index, pd.DatetimeIndex
    ):
        raise ValueError("features and target must have DatetimeIndex indices")
    if features.index.has_duplicates or target.index.has_duplicates:
        raise ValueError("features and target indices must be unique")
    overlap = features.index.intersection(target.index)
    if len(overlap) and len(overlap) != len(features.index.union(target.index)):
        raise ValueError("features and target indices overlap but are not identical")

    aligned = features.join(target.rename("target"), how="inner")
    target_aligned = aligned.pop("target")
    valid = target_aligned.notna() & ~features.reindex(aligned.index).isna().all(axis=1)
    return features.reindex(aligned.index[valid]), target_aligned.loc[valid]


def _split_positions(
    row_count: int,
    train_fraction: float,
    validation_fraction: float,
    test_fraction: float,
    gap_rows: int,
    min_train_rows: int,
) -> tuple[int, int, int, int]:
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(fraction <= 0 or fraction >= 1 for fraction in fractions):
        raise ValueError("split fractions must be in the open interval (0, 1)")
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must sum to 1.0")
    if gap_rows < 0:
        raise ValueError("gap_rows must be non-negative")

    train_end = int(row_count * train_fraction)
    validation_start = train_end + gap_rows
    validation_end = validation_start + int(row_count * validation_fraction)
    test_start = validation_end + gap_rows
    if train_end < min_train_rows or validation_start >= validation_end or test_start >= row_count:
        raise ValueError("not enough rows for the requested time-aware split and gap")
    return train_end, validation_start, validation_end, test_start


def time_aware_split(
    features: pd.DataFrame,
    target: pd.Series | None = None,
    *,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    gap_rows: int = 1,
    min_train_rows: int = 20,
) -> tuple[Any, ...]:
    """Return chronological train/validation/test splits with a purge gap.

    When ``target`` is supplied, six objects are returned: train features,
    validation features, test features, and their corresponding targets.
    Without a target, three feature frames are returned.
    """

    if features.empty:
        empty = features.iloc[0:0]
        return (empty, empty, empty) if target is None else (empty, empty, empty, empty, empty, empty)
    positions = _split_positions(
        len(features),
        train_fraction,
        validation_fraction,
        test_fraction,
        gap_rows,
        min_train_rows,
    )
    train_end, validation_start, validation_end, test_start = positions
    train = features.iloc[:train_end]
    validation = features.iloc[validation_start:validation_end]
    test = features.iloc[test_start:]
    if target is None:
        return train, validation, test

    aligned_features, aligned_target = align_features_target(features, target)
    positions = _split_positions(
        len(aligned_features),
        train_fraction,
        validation_fraction,
        test_fraction,
        gap_rows,
        min_train_rows,
    )
    train_end, validation_start, validation_end, test_start = positions
    return (
        aligned_features.iloc[:train_end],
        aligned_features.iloc[validation_start:validation_end],
        aligned_features.iloc[test_start:],
        aligned_target.iloc[:train_end],
        aligned_target.iloc[validation_start:validation_end],
        aligned_target.iloc[test_start:],
    )


def train_test_split(
    frame: pd.DataFrame,
    test_fraction: float = 0.2,
    *,
    gap_rows: int = 0,
    min_train_rows: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible chronological two-way split."""

    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in the open interval (0, 1).")
    if frame.empty:
        empty = frame.iloc[0:0]
        return empty, empty
    row_count = len(frame)
    train_end = int(row_count * (1.0 - test_fraction))
    test_start = train_end + gap_rows
    if train_end < min_train_rows or test_start >= row_count:
        raise ValueError("not enough rows for the requested split and gap")
    return frame.iloc[:train_end], frame.iloc[test_start:]


@dataclass
class FeaturePreprocessor:
    """Median-impute and standardise features using training statistics only."""

    feature_names: list[str] | None = None
    imputer: SimpleImputer = field(
        default_factory=lambda: SimpleImputer(strategy="median", keep_empty_features=True)
    )
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted_feature_names_: list[str] = field(default_factory=list)

    def fit(self, features: pd.DataFrame) -> FeaturePreprocessor:
        if features.empty:
            raise ValueError("cannot fit preprocessing on an empty feature frame")
        names = self.feature_names or list(features.columns)
        missing = [name for name in names if name not in features.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        matrix = features.loc[:, names].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        self.fitted_feature_names_ = names
        self.imputer.fit(matrix)
        imputed = self.imputer.transform(matrix)
        self.scaler.fit(imputed)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_feature_names_:
            raise ValueError("preprocessor has not been fitted")
        matrix = features.reindex(columns=self.fitted_feature_names_).apply(
            pd.to_numeric, errors="coerce"
        )
        matrix = matrix.replace([np.inf, -np.inf], np.nan)
        transformed = self.scaler.transform(self.imputer.transform(matrix))
        return pd.DataFrame(transformed, index=features.index, columns=self.fitted_feature_names_)

    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        return self.fit(features).transform(features)

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(self.fitted_feature_names_, dtype=object)

    def metadata(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.fitted_feature_names_),
            "imputer_strategy": self.imputer.strategy,
            "scaler": self.scaler.__class__.__name__,
        }


MLPreprocessor = FeaturePreprocessor


def prepare_dataset(
    data: pd.DataFrame,
    *,
    horizon: int = 1,
    threshold: float = 0.0,
    context_frames: Mapping[str, pd.DataFrame] | None = None,
    context_lag: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build causal features and an aligned next-period direction target."""

    features = build_feature_frame(
        data,
        windows=(5, 20, 50),
        context_frames=context_frames,
        context_lag=context_lag,
    )
    target = build_target(data, horizon=horizon, threshold=threshold)
    return align_features_target(features, target)
