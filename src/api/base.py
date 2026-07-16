"""src.api.base — shared infrastructure for all API client sub-modules.

Provides a common logger, a thread-safe per-thread :class:`requests.Session`
for HTTP connection reuse (via :func:`_get_session`), and re-exports the
configuration constants that every API client module needs (API keys,
timeouts, retry counts).

All sub-modules under ``src.api`` should import ``logger``, ``API_TIMEOUT``,
and ``_get_session`` from here rather than re-defining them.

Thread Safety
-------------
:func:`_get_session` returns a thread-local ``requests.Session`` instance.
Each OS thread gets its own session with independent connection pool, so
concurrent calls from different ``ThreadPoolExecutor`` workers never share
TCP connections (avoiding data-races and response interleaving).  Sessions
are lazily created on first access and live as long as the thread exists.
"""

import logging
import threading

import requests

from src.config import (
    GOOGLE_MAPS_API_KEY,
    OPENWEATHER_API_KEY,
    SERPAPI_KEY,
    SERPER_API_KEY,
    API_TIMEOUT,
    API_MAX_RETRIES,
)

logger = logging.getLogger(__name__)

# Per-thread requests.Session instances, lazily created on first access.
# Using ``threading.local()`` ensures each ThreadPoolExecutor worker thread
# owns an independent session object with its own connection pool, so
# concurrent API calls never share or corrupt TCP connections.
_local = threading.local()


def _get_session() -> requests.Session:
    """Return (or create) a thread-local ``requests.Session``.

    Each OS thread gets its own session with independent connection pooling.
    Thread-safe by construction — no locking required.

    Returns:
        requests.Session: Per-thread session with connection reuse.
    """
    s = getattr(_local, "_session", None)
    if s is None:
        s = _local._session = requests.Session()
    return s
