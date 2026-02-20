"""Data Preprocessing Package.

This package contains modules for extracting, processing, and embedding
documents for the Institute Q&A knowledge base.
"""

from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
from institute_qna.data_preprocess.embedding_generation import EmbeddingsGeneration
from institute_qna.data_preprocess.chroma_vector_store import (
    ChromaVectorStoreManager,
    create_embeddings_in_batches,
)

__all__ = [
    "PDFTextExtractor",
    "EmbeddingsGeneration",
    "ChromaVectorStoreManager",
    "KnowledgeBaseCreation",
    "create_embeddings_in_batches",
    "main",
]

try:
    from institute_qna.data_preprocess.azure_ai_search_store import (
        AzureAISearchVectorStoreManager,
        upload_documents_to_azure_ai_search,
    )
    from institute_qna.data_preprocess.vector_store_ingestion import (
        store_documents_in_vector_backend,
        store_documents_in_vector_backend_from_blob,
    )
    __all__.extend([
        "AzureAISearchVectorStoreManager",
        "upload_documents_to_azure_ai_search",
        "store_documents_in_vector_backend",
        "store_documents_in_vector_backend_from_blob",
    ])
except ModuleNotFoundError:
    pass

from institute_qna.data_preprocess.knoweldge_base_creation import (
    KnowledgeBaseCreation,
    main
)
