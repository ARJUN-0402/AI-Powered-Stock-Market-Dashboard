"""Cache helpers for expensive data fetches.

Streamlit's ``@st.cache_data`` decorator requires the module to import
``streamlit``. To keep core services decoupled from Streamlit we wrap the
cache decorator in a helper that fails gracefully when Streamlit is not
available (e.g. during unit tests).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def streamlit_cache(ttl_seconds: int) -> Callable[[F], F]:
    """Return a Streamlit cache decorator that is safe outside Streamlit.

    When Streamlit cannot be imported (for example in unit tests or when this
    module is used in a headless context) the wrapped function is returned
    unchanged.
    """

    def decorator(func: F) -> F:
        try:
            import streamlit as st  # type: ignore

            return st.cache_data(ttl=ttl_seconds)(func)
        except Exception:
            return func

    return decorator
