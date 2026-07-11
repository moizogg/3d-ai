"""Structured logging setup for console (dev) and JSON (cloud) modes."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Literal


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line for cloud aggregation."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        for key in ("stage", "job_id", "provider"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    fmt: Literal["console", "json"] = "console",
    *,
    logger_name: str = "propertyscan",
) -> logging.Logger:
    """Configure the root PropertyScan logger.

    Purpose:
        Provide a single, consistent logging entry point for all stages.

    Inputs:
        level: logging level name (DEBUG, INFO, WARNING, ERROR).
        fmt: ``console`` for human-readable output, ``json`` for machines.

    Outputs:
        Configured ``logging.Logger`` instance.

    Limitations:
        Does not attach file handlers; callers may add them via RunContext artifacts.
    """
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    handler = logging.StreamHandler(sys.stdout)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    logger.addHandler(handler)
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger under the propertyscan namespace."""
    if name is None or name == "propertyscan":
        return logging.getLogger("propertyscan")
    if name.startswith("propertyscan."):
        return logging.getLogger(name)
    return logging.getLogger(f"propertyscan.{name}")
