# institute_qna/logging_config.py
from logging.config import dictConfig
from typing import Optional

class Config:
    DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None) -> None:
    """
    Configure package logging for application use.

    Call this from your app entrypoint (script/notebook) to enable console output.
    Example: institute_qna.logging_config.configure_logging(level="INFO")
    """
    level = (level or "INFO").upper()
    fmt = fmt or Config.DEFAULT_LOG_FORMAT

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {"format": fmt, "datefmt": Config.DEFAULT_DATEFMT},
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": level,
                }
            },
            "root": {
                "handlers": ["console"],
                "level": level,
            },
        }
    )