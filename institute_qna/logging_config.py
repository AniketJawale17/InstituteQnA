import logging
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig
from pathlib import Path
from typing import Optional
from datetime import datetime
import os

try:
    from azure.storage.blob import BlobServiceClient

    AZURE_BLOB_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    AZURE_BLOB_AVAILABLE = False


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class AzureBlobLogHandler(logging.Handler):
    """Logging handler that appends formatted log lines to an Azure Append Blob."""

    def __init__(self, connection_string: str, container_name: str, blob_name: str):
        super().__init__()
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_name = blob_name
        self._append_blob_client: Optional[object] = None
        self._is_emitting = False
        self._ensure_append_blob()

    def _ensure_append_blob(self) -> None:
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(self.container_name)
        if not container_client.exists():
            container_client.create_container()

        self._append_blob_client = container_client.get_blob_client(self.blob_name)
        if not self._append_blob_client.exists():
            self._append_blob_client.create_append_blob()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("azure."):
            return
        if self._is_emitting:
            return

        try:
            self._is_emitting = True
            if self._append_blob_client is None:
                self._ensure_append_blob()

            message = self.format(record)
            payload = (message + "\n").encode("utf-8")
            self._append_blob_client.append_block(payload)
        except Exception:
            self.handleError(record)
        finally:
            self._is_emitting = False


def _build_default_blob_name() -> str:
    run_folder = os.getenv("AZURE_LOGS_RUN_FOLDER") or datetime.now().strftime("%Y%m%d")
    app_name = os.getenv("AZURE_LOG_APP_NAME", "app")
    return f"logs/{run_folder}/{app_name}.log"


def _add_blob_logging_handler(level: str) -> None:
    enabled = os.getenv("ENABLE_BLOB_LOGGING", "false").lower() in {"1", "true", "yes"}
    if not enabled:
        return

    root_logger = logging.getLogger()

    if not AZURE_BLOB_AVAILABLE:
        root_logger.warning("ENABLE_BLOB_LOGGING=true but azure-storage-blob is not installed")
        return

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        root_logger.warning("ENABLE_BLOB_LOGGING=true but AZURE_STORAGE_CONNECTION_STRING is missing")
        return

    container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
    blob_name = os.getenv("AZURE_LOGS_BLOB_NAME", _build_default_blob_name())

    for handler in root_logger.handlers:
        if isinstance(handler, AzureBlobLogHandler) and handler.blob_name == blob_name:
            return

    blob_handler = AzureBlobLogHandler(
        connection_string=connection_string,
        container_name=container_name,
        blob_name=blob_name,
    )
    blob_handler.setLevel(level)
    blob_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATEFMT))
    root_logger.addHandler(blob_handler)


def configure_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    console: Optional[bool] = None,
) -> None:
    """
    Configure logging for applications that use this package.

    - level: logging level string (e.g. "INFO", "DEBUG").
    - log_file: optional path to a file to write logs to (rotating handler).

    Libraries should not call this themselves; application entrypoints should.
    """
    lvl = (level or "INFO").upper()

    console_enabled = console
    if console_enabled is None:
        console_enabled = os.getenv("LOG_TO_CONSOLE", "true").lower() in {"1", "true", "yes"}

    handlers = []
    handler_defs = {}

    if console_enabled:
        handlers.append("console")
        handler_defs["console"] = {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": lvl,
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

    _add_blob_logging_handler(lvl)
