# Agent Notes

## Market data layer

The market data layer lives in `src/data/market_data_service/`.

- `MarketDataService` (`service.py`) is the single entry point. It validates
  inputs, talks to a `MarketDataProvider`, validates the returned frame,
  caches results in a thread-safe TTL cache (`cache.py`) and translates
  provider exceptions into the typed hierarchy in `errors.py`.
- `MarketDataProvider` (`providers/base.py`) is the provider interface. The
  default implementation, `YFinanceProvider`
  (`providers/yfinance_provider.py`), isolates yfinance specifics from the
  application. Add a new provider by subclassing `MarketDataProvider` and
  wiring it via `MarketDataService(provider=...)`.
- DTOs are frozen dataclasses: `Quote` and `MarketStats` (`dto.py`). Both
  expose an `is_available` flag and an `is_delayed` flag. Values are never
  fabricated; missing observations are surfaced as `None` or
  `is_available=False`.
- `validators.py` provides `normalise_ohlcv_frame`, `validate_ohlcv_frame`,
  `require_valid_frame`, `safe_last_close` and `safe_previous_close`.
- `src/data/market_data.py` is a backwards-compat shim used by older
  call sites; it delegates to the service and therefore inherits caching,
  validation and error handling.

### Configuration

`MarketDataServiceConfig` (in `service.py`) exposes `history_ttl_seconds`,
`quote_ttl_seconds`, `stats_ttl_seconds`, `negative_ttl_seconds` and
`min_history_rows`. Use `service.configure(...)` to mutate at runtime or
`service.refresh(symbol=...)` to invalidate cached entries.

### Tests

Deterministic unit tests live in:

- `tests/unit/test_market_data_service.py`
- `tests/unit/test_market_data_validators.py`
- `tests/unit/test_market_data_cache.py`
- `tests/unit/test_yfinance_provider.py`
- `tests/unit/test_market_data_legacy.py`

Run `python -m pytest` and `python -m ruff check src tests app.py`.
