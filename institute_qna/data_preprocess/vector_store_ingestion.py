"""Unified vector store ingestion options for Chroma and Azure AI Search."""

from __future__ import annotations

import json
import os
from typing import List

from langchain_core.documents import Document

from institute_qna.data_preprocess.azure_ai_search_store import upload_documents_to_azure_ai_search
from institute_qna.data_preprocess.chroma_vector_store import create_embeddings_in_batches

try:
    from azure.storage.blob import BlobServiceClient

    AZURE_BLOB_AVAILABLE = True
except ModuleNotFoundError:
    AZURE_BLOB_AVAILABLE = False


def store_documents_in_vector_backend(
    documents: List[Document],
    backend: str = os.getenv("VECTOR_BACKEND", "azure_ai_search"),
    batch_size: int = 100,
    sleep_time: int = 60,
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data"),
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store/ug_admission_data"),
    embedding_model: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    azure_index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "ug-admission-data"),
    azure_vector_dimensions: int = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")),
) -> None:
    """Store documents in selected vector backend.

    Args:
        documents: Documents to store.
        backend: Supported values are "chroma" and "azure_ai_search".
        batch_size: Upload batch size for both backends.
        sleep_time: Pause between Chroma batches.
        chroma_collection_name: Chroma collection name.
        chroma_persist_directory: Chroma persistence path.
        embedding_model: Azure OpenAI embedding model.
        azure_index_name: Azure AI Search index name.
        azure_vector_dimensions: Dimensions matching your embedding model.
    """
    normalized_backend = backend.strip().lower()

    if normalized_backend == "chroma":
        create_embeddings_in_batches(
            documents=documents,
            batch_size=batch_size,
            sleep_time=sleep_time,
            collection_name=chroma_collection_name,
            persist_directory=chroma_persist_directory,
            model=embedding_model,
        )
        return

    if normalized_backend in {"azure", "azure_ai_search", "azure-search"}:
        upload_documents_to_azure_ai_search(
            documents=documents,
            index_name=azure_index_name,
            embedding_model=embedding_model,
            vector_dimensions=azure_vector_dimensions,
            batch_size=batch_size,
        )
        return

    raise ValueError(
        "Unsupported backend. Use 'chroma' or 'azure_ai_search'."
    )


def load_documents_from_blob(
    blob_path: str,
    connection_string: str | None = None,
    container_name: str | None = None,
) -> List[Document]:
    """Load serialized documents from Azure Blob.

    Supported path formats:
    - `azure-blob://<container>/<blob_path>`
    - `<blob_path>` with `container_name` or `AZURE_STORAGE_CONTAINER_NAME`
    """
    if not AZURE_BLOB_AVAILABLE:
        raise ImportError(
            "azure-storage-blob is not installed. Install it with: pip install azure-storage-blob"
        )

    conn = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING is required to load documents from blob")

    parsed_container = container_name or os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
    parsed_blob_path = blob_path

    prefix = "azure-blob://"
    if blob_path.startswith(prefix):
        raw = blob_path[len(prefix):]
        first_slash = raw.find("/")
        if first_slash == -1:
            raise ValueError("Invalid azure-blob path. Expected azure-blob://<container>/<blob_path>")
        parsed_container = raw[:first_slash]
        parsed_blob_path = raw[first_slash + 1 :]

    service = BlobServiceClient.from_connection_string(conn)
    container_client = service.get_container_client(parsed_container)
    blob_client = container_client.get_blob_client(parsed_blob_path)

    payload = blob_client.download_blob().readall().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError(f"Expected list payload in blob {parsed_blob_path}")

    documents = [
        Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {}),
        )
        for item in data
    ]
    return documents


def store_documents_in_vector_backend_from_blob(
    blob_path: str,
    backend: str = os.getenv("VECTOR_BACKEND", "azure_ai_search"),
    batch_size: int = 100,
    sleep_time: int = 60,
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data"),
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store/ug_admission_data"),
    embedding_model: str = os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    azure_index_name: str = os.getenv("AZURE_SEARCH_INDEX_NAME", "ug-admission-data"),
    azure_vector_dimensions: int = int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")),
) -> None:
    """Load processed docs from blob and ingest them into selected vector backend."""
    documents = load_documents_from_blob(blob_path=blob_path)
    store_documents_in_vector_backend(
        documents=documents,
        backend=backend,
        batch_size=batch_size,
        sleep_time=sleep_time,
        chroma_collection_name=chroma_collection_name,
        chroma_persist_directory=chroma_persist_directory,
        embedding_model=embedding_model,
        azure_index_name=azure_index_name,
        azure_vector_dimensions=azure_vector_dimensions,
    )
