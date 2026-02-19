"""RAG Retriever Module for Institute Q&A System.

This module handles document retrieval from either Azure AI Search
or Chroma and prepares context for LLM-based question answering.
"""

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Optional
import json
import logging
import os

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
    from azure.search.documents.models import VectorizedQuery
    AZURE_SEARCH_AVAILABLE = True
except ModuleNotFoundError:
    AZURE_SEARCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Handles retrieval of relevant documents from the vector store."""
    
    def __init__(
        self, 
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        model: Optional[str] = None,
        backend: Optional[str] = None,
        top_k: int = 5
    ):
        """Initialize the RAG Retriever.
        
        Args:
            persist_directory: Path to Chroma vector database (used when backend=chroma)
            collection_name: Collection/index name
            model: Azure OpenAI embedding model name
            backend: Retrieval backend ('azure_ai_search' or 'chroma')
            top_k: Number of documents to retrieve
        """
        self.backend = (backend or os.getenv("RAG_VECTOR_BACKEND", os.getenv("VECTOR_BACKEND", "azure_ai_search"))).strip().lower()
        self.persist_directory = persist_directory or os.getenv(
            "RAG_PERSIST_DIRECTORY",
            os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store/ug_admission_data"),
        )
        self.collection_name = collection_name or os.getenv(
            "RAG_COLLECTION_NAME",
            os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data"),
        )
        model_name = model or os.getenv(
            "RAG_EMBEDDING_MODEL",
            os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        self.top_k = top_k
        self.search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "ug-admission-data")
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=model_name,
        )

        self.vector_store = None
        self.search_client = None

        if self.backend in {"azure", "azure_ai_search", "azure-search"}:
            if not AZURE_SEARCH_AVAILABLE:
                raise ImportError(
                    "azure-search-documents is not installed. "
                    "Install it with: pip install azure-search-documents"
                )

            endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
            api_key = os.getenv("AZURE_SEARCH_API_KEY")
            if not endpoint or not api_key:
                raise ValueError(
                    "AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY must be set for Azure AI Search retrieval"
                )

            self.search_client = SearchClient(
                endpoint=endpoint,
                index_name=self.search_index_name,
                credential=AzureKeyCredential(api_key),
            )
            logger.info("Successfully initialized Azure AI Search retriever on index %s", self.search_index_name)
            return

        # Chroma fallback backend
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info("Successfully loaded Chroma vector store from %s", self.persist_directory)
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise

    def _retrieve_from_azure_search(self, query: str, k: int) -> List[Dict]:
        query_vector = self.embeddings.embed_query(query)
        vector_query = VectorizedQuery(
            vector=query_vector,
            k_nearest_neighbors=k,
            fields="content_vector",
        )

        results = self.search_client.search(
            search_text="*",
            vector_queries=[vector_query],
            top=k,
        )

        formatted_results = []
        for item in results:
            metadata = {}
            metadata_json = item.get("metadata_json")
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json)
                except Exception:
                    metadata = {"metadata_json": str(metadata_json)}

            metadata.setdefault("source", item.get("source", ""))
            metadata.setdefault("title", item.get("title", ""))
            metadata.setdefault("doc_type", item.get("doc_type", ""))
            metadata.setdefault("source_type", item.get("source_type", ""))
            metadata.setdefault("university", item.get("university", ""))
            metadata.setdefault("page", item.get("page", ""))

            formatted_results.append({
                "content": item.get("content", ""),
                "metadata": metadata,
                "score": float(item.get("@search.score", 0.0)),
                "id": item.get("id"),
            })

        return formatted_results
    
    def retrieve(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve (overrides default)
            filter_metadata: Optional metadata filters for retrieval
            
        Returns:
            List of dictionaries containing document content and metadata
        """
        k = top_k or self.top_k
        
        try:
            if self.backend in {"azure", "azure_ai_search", "azure-search"}:
                if filter_metadata:
                    logger.warning("Azure AI Search retriever currently ignores dict filter_metadata")
                formatted_results = self._retrieve_from_azure_search(query, k)
                for entry in formatted_results:
                    entry.pop("score", None)
                    entry.pop("id", None)
            else:
                # Chroma similarity search
                if filter_metadata:
                    results = self.vector_store.similarity_search(
                        query,
                        k=k,
                        filter=filter_metadata
                    )
                else:
                    results = self.vector_store.similarity_search(query, k=k)

                formatted_results = []
                for doc in results:
                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents for query: {query[:50]}...")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise

    def retrieve_test(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Retrieve relevant documents with similarity scores.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries with document content, metadata, and scores
        """
        k = top_k or self.top_k
        
        try:
            if self.backend in {"azure", "azure_ai_search", "azure-search"}:
                formatted_results = self._retrieve_from_azure_search(query, k)
                if score_threshold is not None:
                    formatted_results = [
                        doc for doc in formatted_results
                        if doc.get("score", 0.0) >= score_threshold
                    ]
            else:
                # Chroma similarity search with scores
                results = self.vector_store.similarity_search_with_score(query, k=k)
                formatted_results = []
                for doc, score in results:
                    if score_threshold and score < score_threshold:
                        continue

                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score),
                        "id": doc.id
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents with scores")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {e}")
            raise
       
    def retrieve_with_scores(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """Retrieve relevant documents with similarity scores.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of dictionaries with document content, metadata, and scores
        """
        k = top_k or self.top_k
        
        try:
            if self.backend in {"azure", "azure_ai_search", "azure-search"}:
                formatted_results = self._retrieve_from_azure_search(query, k)
                for entry in formatted_results:
                    entry.pop("id", None)
                if score_threshold is not None:
                    formatted_results = [
                        doc for doc in formatted_results
                        if doc.get("score", 0.0) >= score_threshold
                    ]
            else:
                results = self.vector_store.similarity_search_with_score(query, k=k)

                formatted_results = []
                for doc, score in results:
                    if score_threshold and score < score_threshold:
                        continue

                    formatted_results.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    })
            
            logger.info(f"Retrieved {len(formatted_results)} documents with scores")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error retrieving documents with scores: {e}")
            raise
    
    def get_context_string(self, documents: List[Dict]) -> str:
        """Convert retrieved documents into a formatted context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Source: {source}")
            context_parts.append(f"Content: {content}")
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)
    
    def as_retriever(self, **kwargs):
        """Get a LangChain retriever interface.
        
        Returns:
            LangChain retriever object
        """
        if self.backend in {"azure", "azure_ai_search", "azure-search"}:
            raise NotImplementedError("LangChain retriever adapter is not available for Azure AI Search backend")

        return self.vector_store.as_retriever(
            search_kwargs={"k": kwargs.get("k", self.top_k)}
        )


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize retriever
    retriever = RAGRetriever(top_k=3)
    
    # Test query
    query = "What are the admission requirements for undergraduate programs?"
    
    # Retrieve documents
    docs = retriever.retrieve(query)
    
    print(f"\nQuery: {query}")
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content: {doc['content'][:200]}...")
        print(f"Metadata: {doc['metadata']}")
    
    # Get context string
    context = retriever.get_context_string(docs)
    print(f"\n\n--- Context String ---\n{context[:500]}...")
