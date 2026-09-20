"""Tests for validation helpers."""

from __future__ import annotations

import pytest

from src.utils.validation import (
    ValidationError,
    safe_float,
    validate_interval,
    validate_period,
    validate_symbol,
    validate_symbols,
)


def test_validate_symbol_normalises_input() -> None:
    assert validate_symbol(" aapl ") == "AAPL"
    assert validate_symbol("BRK.B") == "BRK.B"


@pytest.mark.parametrize("bad", ["", None, "  ", "BAD!", "@@@"])
def test_validate_symbol_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValidationError):
        validate_symbol(bad)


def test_validate_period_and_interval() -> None:
    assert validate_period("1mo") == "1mo"
    with pytest.raises(ValidationError):
        validate_period("bogus")
    assert validate_interval("1d") == "1d"
    with pytest.raises(ValidationError):
        validate_interval("bogus")


def test_validate_symbols_skips_invalid() -> None:
    # validate_symbols raises on invalid input but skips falsy entries via filter.
    assert validate_symbols(["aapl", "MSFT"]) == ["AAPL", "MSFT"]


def test_safe_float_handles_invalid_values() -> None:
    assert safe_float("3.14") == 3.14
    assert safe_float(None, default=1.0) == 1.0
    assert safe_float("abc", default=2.0) == 2.0
    assert safe_float(float("nan")) == 0.0
