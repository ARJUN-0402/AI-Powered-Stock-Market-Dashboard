"""Logging utilities."""

from __future__ import annotations

import logging
import sys

from src.config import CONFIG

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DEFAULT_LOGGER_NAME = "stock_dashboard"


def configure_logging(level: str | None = None) -> None:
    """Configure root logging for the application.

    Parameters
    ----------
    level:
        Optional log level override. Falls back to :data:`CONFIG.log_level`.
    """

    log_level = (level or CONFIG.log_level).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger configured for the project."""

    return logging.getLogger(name or _DEFAULT_LOGGER_NAME)
