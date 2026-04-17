"""
aria/core/logger.py — Structured logging setup for ARIA.

Uses loguru to provide:
  - JSON format  : {"timestamp", "level", "component", "message", "data"}
  - Pretty format: colored, human-readable output for development

Usage:
    from aria.core.logger import get_logger
    log = get_logger("llm.client")
    log.info("Request completed", data={"duration_ms": 142})
"""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    from loguru import Logger

# Track whether the global sink has been installed already.
_configured: bool = False


def _json_sink(message: Any) -> None:
    """Write a loguru message as a single-line JSON object to stderr.

    Schema:
        {
            "timestamp": ISO-8601 string,
            "level": "INFO" | "DEBUG" | …,
            "component": str,
            "message": str,
            "data": dict | null
        }

    Args:
        message: A loguru ``Message`` object (str subclass with
            a ``.record`` attribute).
    """
    record = message.record
    payload: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "component": record["extra"].get("component", "aria"),
        "message": record["message"],
        "data": record["extra"].get("data"),
    }
    sys.stderr.write(json.dumps(payload, default=str) + "\n")


_PRETTY_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>[{extra[component]}]</cyan> "
    "{message}\n"
)


def setup_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Configure the global loguru logger.

    Removes default sinks and installs a single stderr sink with the
    requested format.  Safe to call multiple times — subsequent calls
    are no-ops.

    Args:
        level: Minimum log level (TRACE, DEBUG, INFO, WARNING, ERROR).
        fmt: Output format — ``"json"`` or ``"pretty"``.
    """
    global _configured
    if _configured:
        return

    # Remove the default loguru sink.
    logger.remove()

    if fmt == "json":
        logger.add(
            _json_sink,
            level=level.upper(),
            colorize=False,
            serialize=False,
        )
    else:
        logger.add(
            sys.stderr,
            format=_PRETTY_FORMAT,
            level=level.upper(),
            colorize=True,
        )

    _configured = True


def get_logger(component: str) -> Logger:
    """Return a logger instance bound to the given *component* name.

    If global logging has not been configured yet, it will be set up
    with default settings (INFO / json).

    Args:
        component: A dot-separated component identifier,
                   e.g. ``"llm.client"`` or ``"core.config"``.

    Returns:
        A loguru ``Logger`` instance with the *component* extra field.
    """
    if not _configured:
        setup_logging()

    return logger.bind(component=component)
