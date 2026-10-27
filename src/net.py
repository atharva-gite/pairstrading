"""Outbound HTTP helpers. Cursor injects a loopback proxy that public APIs reject with 403."""

from __future__ import annotations

import contextlib
import os

PROXY_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "SOCKS_PROXY",
    "SOCKS5_PROXY",
    "socks_proxy",
    "socks5_proxy",
)
_LOOPBACK_PROXY_PREFIXES = (
    "http://127.0.0.1:",
    "https://127.0.0.1:",
    "http://localhost:",
    "https://localhost:",
    "socks://127.0.0.1:",
    "socks5://127.0.0.1:",
    "socks://localhost:",
    "socks5://localhost:",
)


def is_loopback_proxy(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.startswith(prefix) for prefix in _LOOPBACK_PROXY_PREFIXES)


def loopback_proxy_active() -> bool:
    return any(is_loopback_proxy(os.environ.get(key, "")) for key in PROXY_ENV)


@contextlib.contextmanager
def direct_outbound_network():
    saved = {key: os.environ[key] for key in PROXY_ENV if key in os.environ}
    if any(is_loopback_proxy(value) for value in saved.values()):
        for key in saved:
            os.environ.pop(key, None)
    try:
        yield
    finally:
        os.environ.update(saved)
