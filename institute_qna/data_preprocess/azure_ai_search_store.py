"""Azure AI Search vector index and document upload module."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        HnswAlgorithmConfiguration,
        SearchField,
        SearchFieldDataType,
        SearchIndex,
        SearchableField,
        SimpleField,
        VectorSearch,
        VectorSearchProfile,
    )
    AZURE_SEARCH_AVAILABLE = True
except ModuleNotFoundError:
    AZURE_SEARCH_AVAILABLE = False
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)


class AzureAISearchVectorStoreManager:
    """Creates Azure AI Search index and uploads vectorized documents."""

    def __init__(
        self,
        index_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        vector_dimensions: Optional[int] = None,
    ):
        if not AZURE_SEARCH_AVAILABLE:
            raise ImportError(
                "azure-search-documents is not installed. "
                "Install it with: pip install azure-search-documents"
            )

        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not self.search_endpoint:
            raise ValueError("AZURE_SEARCH_ENDPOINT environment variable is not set.")
        if not self.search_api_key:
            raise ValueError("AZURE_SEARCH_API_KEY environment variable is not set.")
        if not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")
        if not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")

        self.index_name = index_name or os.getenv("AZURE_SEARCH_INDEX_NAME", "ug-admission-data")
        self.embedding_model = embedding_model or os.getenv(
            "AZURE_OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        )
        self.vector_dimensions = vector_dimensions or int(
            os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")
        )

        credential = AzureKeyCredential(self.search_api_key)
        self.index_client = SearchIndexClient(endpoint=self.search_endpoint, credential=credential)
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=credential,
        )
        self.embeddings = AzureOpenAIEmbeddings(
            model=self.embedding_model,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            api_version=self.azure_openai_api_version,
        )

    def create_or_update_index(self) -> None:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchableField(name="source", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchableField(name="title", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="university", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="source_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="doc_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="page", type=SearchFieldDataType.String, filterable=True, sortable=True),
            SearchableField(name="metadata_json", type=SearchFieldDataType.String),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=self.vector_dimensions,
                vector_search_profile_name="default-vector-profile",
            ),
        ]

        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="default-vector-profile",
                    algorithm_configuration_name="default-hnsw",
                )
            ],
            algorithms=[HnswAlgorithmConfiguration(name="default-hnsw")],
        )

        index = SearchIndex(
            name=self.index_name,
            fields=fields,
            vector_search=vector_search,
        )

        self.index_client.create_or_update_index(index)
        logger.info("Azure AI Search index ready: %s", self.index_name)

    def _metadata_to_json(self, metadata: Dict[str, Any]) -> str:
        try:
            return json.dumps(metadata, ensure_ascii=False)
        except Exception:
            safe_metadata = {k: str(v) for k, v in metadata.items()}
            return json.dumps(safe_metadata, ensure_ascii=False)

    def _stable_doc_id(self, document: Document, index: int) -> str:
        metadata = document.metadata or {}
        source = str(metadata.get("source", ""))
        page = str(metadata.get("page", ""))
        raw = f"{source}|{page}|{index}|{document.page_content[:128]}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:64]

    def _document_to_search_payload(self, document: Document, index: int) -> Dict[str, Any]:
        metadata = document.metadata or {}
        content = document.page_content or ""
        vector = self.embeddings.embed_query(content)

        return {
            "id": self._stable_doc_id(document, index),
            "content": content,
            "source": str(metadata.get("source", "")),
            "title": str(metadata.get("title", "")),
            "university": str(metadata.get("university", "")),
            "source_type": str(metadata.get("source_type", "")),
            "doc_type": str(metadata.get("doc_type", "")),
            "page": str(metadata.get("page", "")),
            "metadata_json": self._metadata_to_json(metadata),
            "content_vector": vector,
        }

    def upload_documents(self, documents: List[Document], batch_size: int = 100) -> None:
        if not documents:
            logger.warning("No documents provided for Azure AI Search upload")
            return

        self.create_or_update_index()

        payloads = [self._document_to_search_payload(doc, idx) for idx, doc in enumerate(documents)]

        for start_idx in range(0, len(payloads), batch_size):
            batch = payloads[start_idx:start_idx + batch_size]
            results = self.search_client.upload_documents(documents=batch)
            failed = [result.key for result in results if not result.succeeded]
            if failed:
                raise RuntimeError(
                    f"Failed to upload {len(failed)} documents to Azure AI Search: {failed[:5]}"
                )
            logger.info(
                "Uploaded Azure AI Search batch: %s to %s",
                start_idx,
                min(start_idx + batch_size, len(payloads)),
            )

        logger.info("Uploaded %s documents to Azure AI Search index %s", len(payloads), self.index_name)


def upload_documents_to_azure_ai_search(
    documents: List[Document],
    index_name: Optional[str] = None,
    embedding_model: Optional[str] = None,
    vector_dimensions: Optional[int] = None,
    batch_size: int = 100,
) -> None:
    """Convenience function to create index and upload all documents."""
    manager = AzureAISearchVectorStoreManager(
        index_name=index_name,
        embedding_model=embedding_model,
        vector_dimensions=vector_dimensions,
    )
    manager.upload_documents(documents=documents, batch_size=batch_size)
