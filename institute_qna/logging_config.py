import logging
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None, log_file: Optional[str] = None) -> None:
    """
    Configure logging for applications that use this package.

    - level: logging level string (e.g. "INFO", "DEBUG").
    - log_file: optional path to a file to write logs to (rotating handler).

    Libraries should not call this themselves; application entrypoints should.
    """
    lvl = (level or "INFO").upper()

    handlers = ["console"]
    handler_defs = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": lvl,
        }
    }

    if log_file:
        # ensure parent directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append("file")
        handler_defs["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,
            "backupCount": 3,
            "formatter": "default",
            "level": lvl,
        }

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {"default": {"format": DEFAULT_LOG_FORMAT, "datefmt": DEFAULT_DATEFMT}},
            "handlers": handler_defs,
            "root": {"handlers": handlers, "level": lvl},
        }
    )
