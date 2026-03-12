import logging
from logging.handlers import RotatingFileHandler
from logging.config import dictConfig
from pathlib import Path
from typing import Optional
from datetime import datetime
import os
import uuid

try:
    from azure.data.tables import TableServiceClient

    AZURE_TABLES_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    AZURE_TABLES_AVAILABLE = False


DEFAULT_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


class AzureTableLogHandler(logging.Handler):
    """Logging handler that writes each formatted log record to Azure Table Storage."""

    def __init__(self, connection_string: str, table_name: str, partition_key: str):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self.partition_key = partition_key
        self._table_client: Optional[object] = None
        self._is_emitting = False
        self._ensure_table()

    def _ensure_table(self) -> None:
        table_service_client = TableServiceClient.from_connection_string(self.connection_string)
        table_service_client.create_table_if_not_exists(self.table_name)
        self._table_client = table_service_client.get_table_client(self.table_name)

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("azure."):
            return
        if self._is_emitting:
            return

        try:
            self._is_emitting = True
            if self._table_client is None:
                self._ensure_table()

            message = self.format(record)
            created_at = datetime.utcnow().isoformat(timespec="seconds") + "Z"
            row_key = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex}"
            entity = {
                "PartitionKey": self.partition_key,
                "RowKey": row_key,
                "LoggedAtUtc": created_at,
                "LoggerName": record.name,
                "Level": record.levelname,
                "Message": message,
                "Module": record.module,
                "FunctionName": record.funcName,
                "LineNumber": record.lineno,
            }
            if record.exc_info:
                entity["Exception"] = self.formatter.formatException(record.exc_info)

            self._table_client.create_entity(entity=entity)
        except Exception:
            self.handleError(record)
        finally:
            self._is_emitting = False


def _build_default_logs_table_name() -> str:
    return os.getenv("AZURE_LOGS_TABLE_NAME", "AppLogs")


def _add_table_logging_handler(level: str) -> None:
    enabled = os.getenv("ENABLE_TABLE_LOGGING", os.getenv("ENABLE_BLOB_LOGGING", "false")).lower() in {"1", "true", "yes"}
    if not enabled:
        return

    root_logger = logging.getLogger()

    if not AZURE_TABLES_AVAILABLE:
        root_logger.warning("Table logging enabled but azure-data-tables is not installed")
        return

    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        root_logger.warning("Table logging enabled but AZURE_STORAGE_CONNECTION_STRING is missing")
        return

    table_name = _build_default_logs_table_name()
    partition_key = os.getenv("AZURE_LOGS_PARTITION_KEY", os.getenv("AZURE_LOG_APP_NAME", "InstituteQnA"))

    for handler in root_logger.handlers:
        if isinstance(handler, AzureTableLogHandler) and handler.table_name == table_name:
            return

    try:
        table_handler = AzureTableLogHandler(
            connection_string=connection_string,
            table_name=table_name,
            partition_key=partition_key,
        )
        table_handler.setLevel(level)
        table_handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT, DEFAULT_DATEFMT))
        root_logger.addHandler(table_handler)
    except Exception as e:
        root_logger.warning("Failed to initialize Azure Table logging handler: %s", e)


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

    azure_http_level = os.getenv("AZURE_HTTP_LOG_LEVEL", "WARNING").upper()
    logging.getLogger("azure").setLevel(azure_http_level)
    logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(azure_http_level)

    _add_table_logging_handler(lvl)
