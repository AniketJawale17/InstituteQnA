import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import logging

logger = logging.getLogger(__name__)

try:
    from azure.data.tables import TableServiceClient  # type: ignore[import-not-found]

    AZURE_TABLES_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    AZURE_TABLES_AVAILABLE = False


class QueryAuditTableStore:
    """Persist and read query telemetry entities from Azure Table Storage."""

    def __init__(self) -> None:
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        self.table_name = os.getenv("AZURE_QUERY_AUDIT_TABLE_NAME", "QueryAuditLogs")
        self.partition_key = os.getenv("AZURE_QUERY_AUDIT_PARTITION_KEY", "query-audit")
        self._table_client = None

    def _get_table_client(self):
        if not AZURE_TABLES_AVAILABLE:
            raise RuntimeError("azure-data-tables is not installed")
        if not self.connection_string:
            raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING is missing")

        if self._table_client is None:
            service = TableServiceClient.from_connection_string(self.connection_string)
            service.create_table_if_not_exists(self.table_name)
            self._table_client = service.get_table_client(self.table_name)
        return self._table_client

    @staticmethod
    def _to_text(value: Any, max_length: int) -> str:
        text = "" if value is None else str(value)
        if len(text) > max_length:
            return text[: max_length - 3] + "..."
        return text

    def save_query(
        self,
        query: str,
        answer: str,
        context: str,
        status_code: int,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        table_client = self._get_table_client()
        now_utc = datetime.now(timezone.utc)
        row_key = f"{now_utc.strftime('%Y%m%d%H%M%S%f')}-{uuid.uuid4().hex}"
        meta = metadata or {}

        entity = {
            "PartitionKey": self.partition_key,
            "RowKey": row_key,
            "Query": self._to_text(query, 12000),
            "Answer": self._to_text(answer, 30000),
            "Context": self._to_text(context, 30000),
            "StatusCode": int(status_code),
            "Success": bool(status_code < 400),
            "CreatedAtUtc": now_utc.isoformat(),
            "ProcessingMs": int(meta.get("processing_ms", 0) or 0),
            "NumSources": int(meta.get("num_sources", 0) or 0),
            "TopK": int(meta.get("top_k", 0) or 0),
            "ReturnSources": bool(meta.get("return_sources", False)),
            "Endpoint": self._to_text(meta.get("endpoint", "/query"), 120),
            "HttpMethod": self._to_text(meta.get("http_method", "POST"), 16),
            "RequestId": self._to_text(meta.get("request_id", ""), 200),
            "ClientIp": self._to_text(meta.get("client_ip", ""), 100),
            "UserAgent": self._to_text(meta.get("user_agent", ""), 1500),
            "ModelProvider": self._to_text(meta.get("model_provider", ""), 100),
            "ModelName": self._to_text(meta.get("model_name", ""), 200),
            "Environment": self._to_text(meta.get("environment", ""), 100),
            "HostName": self._to_text(meta.get("host_name", ""), 255),
            "ProcessId": int(meta.get("process_id", 0) or 0),
            "AppVersion": self._to_text(meta.get("app_version", ""), 64),
            "ErrorMessage": self._to_text(meta.get("error_message", ""), 8000),
            "HttpReferrer": self._to_text(meta.get("http_referrer", ""), 1024),
        }
        table_client.create_entity(entity=entity)

    def fetch_recent_queries(self, limit: int = 200) -> List[Dict[str, Any]]:
        table_client = self._get_table_client()
        entities = list(table_client.list_entities(results_per_page=max(20, min(limit, 1000))))

        records: List[Dict[str, Any]] = []
        for item in entities:
            records.append(
                {
                    "query": item.get("Query", ""),
                    "answer": item.get("Answer", ""),
                    "context": item.get("Context", ""),
                    "created_at": item.get("CreatedAtUtc", ""),
                    "status_code": item.get("StatusCode", 0),
                    "success": item.get("Success", False),
                    "processing_ms": item.get("ProcessingMs", 0),
                    "num_sources": item.get("NumSources", 0),
                    "top_k": item.get("TopK", 0),
                    "return_sources": item.get("ReturnSources", False),
                    "endpoint": item.get("Endpoint", ""),
                    "http_method": item.get("HttpMethod", ""),
                    "request_id": item.get("RequestId", ""),
                    "client_ip": item.get("ClientIp", ""),
                    "user_agent": item.get("UserAgent", ""),
                    "model_provider": item.get("ModelProvider", ""),
                    "model_name": item.get("ModelName", ""),
                    "environment": item.get("Environment", ""),
                    "host_name": item.get("HostName", ""),
                    "process_id": item.get("ProcessId", 0),
                    "app_version": item.get("AppVersion", ""),
                    "error_message": item.get("ErrorMessage", ""),
                    "http_referrer": item.get("HttpReferrer", ""),
                }
            )

        records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return records[:limit]


def build_context_summary(response_payload: Dict[str, Any], max_chars: int = 4000) -> str:
    """Create a compact context string from RAG response sources for persistence."""
    sources = response_payload.get("sources") or []
    if not isinstance(sources, list) or not sources:
        return ""

    chunks: List[str] = []
    for index, source in enumerate(sources[:5], start=1):
        metadata = source.get("metadata") or {}
        title = metadata.get("source") or f"source-{index}"
        snippet = (source.get("content") or "").strip()
        if snippet:
            chunks.append(f"[{index}] {title}: {snippet}")

    context_text = "\n".join(chunks).strip()
    if len(context_text) > max_chars:
        return context_text[:max_chars] + "..."
    return context_text
