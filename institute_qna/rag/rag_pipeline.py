"""RAG Pipeline Module for Institute Q&A System.

This module combines retrieval and generation into a complete RAG pipeline.
"""

from institute_qna.rag.retriever import RAGRetriever
from institute_qna.rag.llm_handler import LLMHandler
from typing import Dict, List, Optional
import logging
import os
import time

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline combining retrieval and generation."""
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        top_k: int = 5,
        temperature: float = 0.3,
        system_prompt: Optional[str] = None
    ):
        """Initialize the RAG Pipeline.
        
        Args:
            persist_directory: Path to Chroma vector database
            collection_name: Name of the Chroma collection
            embedding_model: Embedding model name
            llm_provider: LLM provider ("azure" or "google")
            llm_model: LLM model name
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            system_prompt: Custom system prompt
        """
        persist_directory = persist_directory or os.getenv(
            "RAG_PERSIST_DIRECTORY",
            os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store/ug_admission_data"),
        )
        collection_name = collection_name or os.getenv(
            "RAG_COLLECTION_NAME",
            os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data"),
        )
        embedding_model = embedding_model or os.getenv(
            "RAG_EMBEDDING_MODEL",
            os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        llm_provider = llm_provider or os.getenv("LLM_PROVIDER", "google")

        # Initialize retriever
        self.retriever = RAGRetriever(
            persist_directory=persist_directory,
            collection_name=collection_name,
            model=embedding_model,
            backend=os.getenv("RAG_VECTOR_BACKEND", os.getenv("VECTOR_BACKEND", "azure_ai_search")),
            top_k=top_k
        )
        
        # Initialize LLM handler
        self.llm_handler = LLMHandler(
            provider=llm_provider,
            model=llm_model,
            temperature=temperature,
            system_prompt=system_prompt
        )
        
        logger.info("RAG Pipeline initialized successfully")
    
    def query(
        self, 
        question: str,
        top_k: Optional[int] = None,
        return_sources: bool = True,
        stream: bool = False
    ) -> Dict:
        """Process a question through the RAG pipeline.
        
        Args:
            question: User's question
            top_k: Number of documents to retrieve (overrides default)
            return_sources: Whether to include source documents in response
            stream: Whether to stream the LLM response
            
        Returns:
            Dictionary containing answer and optionally sources
        """
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"Processing query: {question}")
            retrieved_docs = self.retriever.retrieve(question, top_k=top_k)
            logger.info(f"Retrieved documents: {retrieved_docs}")
            
            if not retrieved_docs:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Step 2: Format context
            context = self.retriever.get_context_string(retrieved_docs)
            
            # Step 3: Generate answer
            answer = self.llm_handler.generate_answer(
                question=question,
                context=context,
                stream=stream
            )
            
            # Prepare response
            response = {
                "question": question,
                "answer": answer,
                "processing_time": round(time.time() - start_time, 2)
            }
            
            # Include sources if requested
            if return_sources:
                response["sources"] = [
                    {
                        "content": doc["content"][:300] + "..." if len(doc["content"]) > 300 else doc["content"],
                        "metadata": doc["metadata"]
                    }
                    for doc in retrieved_docs
                ]
                response["num_sources"] = len(retrieved_docs)
            
            logger.info(f"Query processed in {response['processing_time']} seconds")
            return response
        
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "question": question,
                "answer": "I encountered an error while processing your question. Please try again.",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def batch_query(
        self,
        questions: List[str],
        top_k: Optional[int] = None,
        return_sources: bool = False
    ) -> List[Dict]:
        """Process multiple questions in batch.
        
        Args:
            questions: List of questions
            top_k: Number of documents to retrieve per question
            return_sources: Whether to include sources
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} questions")
        
        responses = []
        for question in questions:
            response = self.query(
                question=question,
                top_k=top_k,
                return_sources=return_sources
            )
            responses.append(response)
        
        return responses
    
    def query_with_filter(
        self,
        question: str,
        metadata_filter: Dict,
        top_k: Optional[int] = None,
        return_sources: bool = True
    ) -> Dict:
        """Query with metadata filtering.
        
        Args:
            question: User's question
            metadata_filter: Dictionary of metadata filters
            top_k: Number of documents to retrieve
            return_sources: Whether to include sources
            
        Returns:
            Response dictionary
        """
        start_time = time.time()
        
        try:
            # Retrieve with filter
            retrieved_docs = self.retriever.retrieve(
                question, 
                top_k=top_k,
                filter_metadata=metadata_filter
            )
            
            if not retrieved_docs:
                return {
                    "question": question,
                    "answer": "I couldn't find any relevant information matching your criteria.",
                    "sources": [],
                    "processing_time": time.time() - start_time
                }
            
            # Generate answer
            context = self.retriever.get_context_string(retrieved_docs)
            answer = self.llm_handler.generate_answer(question, context)
            
            response = {
                "question": question,
                "answer": answer,
                "filter_applied": metadata_filter,
                "processing_time": round(time.time() - start_time, 2)
            }
            
            if return_sources:
                response["sources"] = [
                    {"content": doc["content"][:300] + "...", "metadata": doc["metadata"]}
                    for doc in retrieved_docs
                ]
            
            return response
        
        except Exception as e:
            logger.error(f"Error in filtered query: {e}")
            return {
                "question": question,
                "answer": "Error processing query with filters.",
                "error": str(e)
            }
    
    def update_prompt(self, new_prompt: str):
        """Update the system prompt.
        
        Args:
            new_prompt: New prompt template
        """
        self.llm_handler.update_system_prompt(new_prompt)
        logger.info("System prompt updated")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        llm_provider="azure",
        top_k=3,
        temperature=0.2
    )
    
    # Test queries
    questions = [
        "What are the admission requirements for B.Tech programs?",
        "How many departments are there in COEP?",
        "What is the fee structure for undergraduate programs?"
    ]
    
    print("=" * 80)
    print("RAG PIPELINE DEMO")
    print("=" * 80)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Question {i}]: {question}")
        print("-" * 80)
        
        response = pipeline.query(question, return_sources=True)
        
        print(f"\n[Answer]:\n{response['answer']}")
        print(f"\n[Processing Time]: {response['processing_time']} seconds")
        print(f"[Sources Used]: {response.get('num_sources', 0)}")
        
        if response.get('sources'):
            print("\n[Source Preview]:")
            for j, source in enumerate(response['sources'][:2], 1):
                print(f"  {j}. {source['content'][:100]}...")
        
        print("=" * 80)
