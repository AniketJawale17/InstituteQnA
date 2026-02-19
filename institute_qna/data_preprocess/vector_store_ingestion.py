"""Unified vector store ingestion options for Chroma and Azure AI Search."""

from __future__ import annotations

import os
from typing import List

from langchain_core.documents import Document

from institute_qna.data_preprocess.azure_ai_search_store import upload_documents_to_azure_ai_search
from institute_qna.data_preprocess.chroma_vector_store import create_embeddings_in_batches


def store_documents_in_vector_backend(
    documents: List[Document],
    backend: str = os.getenv("VECTOR_BACKEND", "chroma"),
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
