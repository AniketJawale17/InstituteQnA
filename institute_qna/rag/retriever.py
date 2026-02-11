"""RAG Retriever Module for Institute Q&A System.

This module handles document retrieval from the Chroma vector database
and prepares context for LLM-based question answering.
"""

from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RAGRetriever:
    """Handles retrieval of relevant documents from the vector store."""
    
    def __init__(
        self, 
        persist_directory: str = "./ug_admission_data",
        collection_name: str = "UG_admission_data",
        model: str = "text-embedding-3-small",
        top_k: int = 5
    ):
        """Initialize the RAG Retriever.
        
        Args:
            persist_directory: Path to the Chroma vector database
            collection_name: Name of the collection in Chroma
            model: Azure OpenAI embedding model name
            top_k: Number of documents to retrieve
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.top_k = top_k
        
        # Initialize embeddings
        self.embeddings = AzureOpenAIEmbeddings(
            model=model,
        )
        
        # Load vector store
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )
            logger.info(f"Successfully loaded vector store from {persist_directory}")
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
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
            # Perform similarity search
            if filter_metadata:
                results = self.vector_store.similarity_search(
                    query, 
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            # print(results)
            # Format results
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
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            # print(results)
            # Format results
            formatted_results = []
            for doc, score in results:
                # Skip if score is below threshold
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
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                # Skip if score is below threshold
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
