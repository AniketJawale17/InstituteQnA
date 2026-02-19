"""Chroma vector store ingestion module."""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from time import sleep
from typing import List, Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()
logger = logging.getLogger(__name__)


class ChromaVectorStoreManager:
    """Creates and updates a Chroma vector store using Azure OpenAI embeddings."""

    def __init__(
        self,
        model: Optional[str] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not self.azure_openai_api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is not set.")
        if not self.azure_openai_endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is not set.")

        self.model = model or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data")
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "./vector_store/ug_admission_data",
        )
        self.vector_store: Optional[Chroma] = None

    def _embeddings(self) -> AzureOpenAIEmbeddings:
        return AzureOpenAIEmbeddings(
            model=self.model,
            azure_endpoint=self.azure_openai_endpoint,
            api_key=self.azure_openai_api_key,
            api_version=self.azure_openai_api_version,
        )

    def create_or_load_store(self) -> Chroma:
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings(),
            persist_directory=self.persist_directory,
        )
        return self.vector_store

    def add_documents(self, docs: List[Document]) -> Chroma:
        if not docs:
            raise ValueError("docs parameter cannot be empty")

        if self.vector_store is None:
            self.create_or_load_store()

        normalized_docs = filter_complex_metadata(docs)
        unique_ids = [str(uuid.uuid4()) for _ in range(len(normalized_docs))]
        self.vector_store.add_documents(documents=normalized_docs, ids=unique_ids)
        logger.info("Added %s documents to Chroma", len(normalized_docs))
        return self.vector_store

    def load_existing_store(self) -> Chroma:
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(f"Vector store not found at {self.persist_directory}")
        return self.create_or_load_store()


class EmbeddingsGeneration(ChromaVectorStoreManager):
    """Backward-compatible wrapper for existing imports."""

    def openai_embeddings_generation(
        self,
        docs: List[Document],
        model: Optional[str] = None,
    ) -> Chroma:
        if model:
            self.model = model
        self.create_or_load_store()
        return self.add_documents(docs)

    def get_vector_store(self) -> Optional[Chroma]:
        return self.vector_store

    def load_existing_vector_store(self) -> Chroma:
        return self.load_existing_store()


def create_embeddings_in_batches(
    documents: List[Document],
    batch_size: int = 70,
    sleep_time: int = 60,
    collection_name: Optional[str] = None,
    persist_directory: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """Create Chroma embeddings in batches to avoid rate limits."""
    if not documents:
        logger.warning("No documents provided for Chroma embedding generation")
        return

    logger.info("Creating Chroma embeddings for %s documents", len(documents))

    manager = EmbeddingsGeneration(
        collection_name=collection_name or os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data"),
        persist_directory=persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            "./vector_store/ug_admission_data",
        ),
        model=model or os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    first_batch_size = min(batch_size, len(documents))
    manager.openai_embeddings_generation(docs=documents[:first_batch_size])
    logger.info("First Chroma batch complete (%s docs)", first_batch_size)

    if len(documents) <= batch_size:
        logger.info("All Chroma embeddings created successfully")
        return

    for start_idx in range(batch_size, len(documents), batch_size):
        end_idx = min(start_idx + batch_size, len(documents))
        logger.info("Sleeping %s seconds before next Chroma batch", sleep_time)
        sleep(sleep_time)
        manager.add_documents(documents[start_idx:end_idx])
        logger.info("Chroma batch complete: docs %s to %s", start_idx, end_idx)

    logger.info("All Chroma embeddings created successfully")
