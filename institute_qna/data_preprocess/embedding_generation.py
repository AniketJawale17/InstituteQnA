"""This module contains the Embeddings Generation class and methods"""

from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from typing import Optional, List
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class EmbeddingsGeneration:
    """Embedding Generation Class for creating and managing vector embeddings.
    
    This class handles the generation of embeddings using Azure OpenAI and stores
    them in a Chroma vector database for efficient similarity search.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        collection_name: str = "UG_admission_data",
        persist_directory: str = "./ug_admission_data"
    ):
        """Initialize the Embeddings Generation class.
        
        Args:
            model: Name of the Azure OpenAI embedding model
            collection_name: Name of the Chroma collection
            persist_directory: Directory to persist the vector store
            
        Raises:
            ValueError: If required environment variables are not set
        """
        # Load from environment variables for security
        self.AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
        self.AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        
        # Validate required environment variables
        if not self.AZURE_OPENAI_API_KEY:
            raise ValueError(
                "AZURE_OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file."
            )
        if not self.AZURE_OPENAI_ENDPOINT:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT environment variable is not set. "
                "Please set it in your .env file."
            )
        
        self.model = model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.vector_store: Optional[Chroma] = None
        
        logger.info(
            f"Initialized EmbeddingsGeneration with model: {model}, "
            f"collection: {collection_name}"
        )


    def openai_embeddings_generation(
            self,
            docs: List,
            model: Optional[str] = None,
        ) -> Chroma:
        """Generate OpenAI embeddings and create/initialize vector store.

        Args:
            docs: List of Document objects to embed and store
            model: Embedding model name (uses instance model if not provided)
            
        Returns:
            Chroma vector store instance
            
        Raises:
            ValueError: If docs is empty or invalid
            RuntimeError: If embedding generation fails
        """
        if not docs:
            raise ValueError("docs parameter cannot be empty")
        
        if not isinstance(docs, list):
            docs = [docs]
        
        model = model or self.model
        logger.info(f"Generating embeddings for {len(docs)} documents using {model}")

        try:
            # Initialize Azure OpenAI Embedding model
            embeddings = AzureOpenAIEmbeddings(
                model=model,
                azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
                api_key=self.AZURE_OPENAI_API_KEY,
                api_version=self.AZURE_OPENAI_API_VERSION
            )

            # Create or load vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
            
            # Add documents with progress tracking
            logger.info(f"Adding {len(docs)} documents to vector store...")
            self.vector_store.add_documents(documents=docs)
            logger.info(f"Successfully added {len(docs)} documents to vector store")

            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise RuntimeError(f"Embedding generation failed: {e}") from e
    
    def add_documents(self, docs: List) -> Chroma:
        """Add documents to the existing vector store.
        
        Args:
            docs: List of Document objects to add
            
        Returns:
            Updated Chroma vector store
            
        Raises:
            ValueError: If vector store is not initialized or docs is empty
        """
        if self.vector_store is None:
            raise ValueError(
                "Vector store not initialized. "
                "Please run openai_embeddings_generation first."
            )
        
        if not docs:
            raise ValueError("docs parameter cannot be empty")
        
        if not isinstance(docs, list):
            docs = [docs]
        
        logger.info(f"Adding {len(docs)} documents to existing vector store...")
        
        try:
            self.vector_store.add_documents(documents=docs)
            logger.info(f"Successfully added {len(docs)} documents")
            return self.vector_store
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def get_vector_store(self) -> Optional[Chroma]:
        """Get the current vector store instance.
        
        Returns:
            Chroma vector store or None if not initialized
        """
        return self.vector_store
    
    def load_existing_vector_store(self) -> Chroma:
        """Load an existing vector store from disk.
        
        Returns:
            Loaded Chroma vector store
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        from pathlib import Path
        
        if not Path(self.persist_directory).exists():
            raise FileNotFoundError(
                f"Vector store not found at {self.persist_directory}"
            )
        
        logger.info(f"Loading existing vector store from {self.persist_directory}")
        
        try:
            embeddings = AzureOpenAIEmbeddings(
                model=self.model,
                azure_endpoint=self.AZURE_OPENAI_ENDPOINT,
                api_key=self.AZURE_OPENAI_API_KEY,
                api_version=self.AZURE_OPENAI_API_VERSION
            )
            
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=embeddings,
                persist_directory=self.persist_directory,
            )
            
            logger.info("Successfully loaded existing vector store")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise