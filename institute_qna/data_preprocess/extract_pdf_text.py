"""PDF Text Extraction Module.

This module provides utilities to extract text from PDF files and split them
into manageable chunks for processing and embedding generation.
"""

from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import logging

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Utility class to extract text from PDF files and split into chunks.
    
    This class handles both text-based and hybrid PDFs, providing methods to
    extract and chunk text content for downstream processing.
    """

    def __init__(
              self, 
              pdf_path: str = "attachments/brochure.pdf",
              chunk_size: int = 2000, 
              chunk_overlap: int = 200
            ):
        """Initialize the PDF Text Extractor.
        
        Args:
            pdf_path: Path to the PDF file to extract
            chunk_size: Size of text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.extracted_text: List[Document] = []

    def extract_text_from_text_pdf(self) -> List[Document]:
        """Extract text from a text-based PDF file and split into chunks.
        
        Returns:
            List of Document objects containing chunked text
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If PDF extraction fails
        """
        path = Path(self.pdf_path)
        if not path.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        try:
            logger.info(f"Loading PDF: {self.pdf_path}")
            loader = PyPDFLoader(str(path))
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages from PDF")
            
        except Exception as e:
            logger.error(f"Failed to load PDF {self.pdf_path}: {e}")
            raise RuntimeError(f"PDF extraction failed: {e}") from e
        
        # Split pages into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.extracted_text = splitter.split_documents(pages)
        logger.info(f"✅ Split {self.pdf_path} into {len(self.extracted_text)} text chunks")

        return self.extracted_text
    
    def extract_multiple_pdfs(self, pdf_folder_path: str) -> List[Document]:
        """Extract text from multiple PDF files in a folder.
        
        Args:
            pdf_folder_path: Path to folder containing PDF files
            
        Returns:
            List of all chunked documents from all PDFs
            
        Raises:
            FileNotFoundError: If folder doesn't exist
            ValueError: If no PDF files found in folder
        """
        folder_path = Path(pdf_folder_path)
        
        if not folder_path.exists():
            logger.error(f"Folder not found: {pdf_folder_path}")
            raise FileNotFoundError(f"Folder not found: {pdf_folder_path}")
        
        pdf_paths = list(folder_path.glob("*.pdf"))
        
        if not pdf_paths:
            logger.warning(f"No PDF files found in {pdf_folder_path}")
            raise ValueError(f"No PDF files found in {pdf_folder_path}")
        
        logger.info(f"Found {len(pdf_paths)} PDF files to process")
        
        all_chunks = []
        successful = 0
        failed = 0
        
        for pdf_path in pdf_paths:
            try:
                self.pdf_path = str(pdf_path)
                chunks = self.extract_text_from_text_pdf()
                all_chunks.extend(chunks)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                failed += 1
                continue
        
        logger.info(
            f"Completed PDF extraction: {successful} successful, "
            f"{failed} failed, {len(all_chunks)} total chunks"
        )
        
        return all_chunks
    
    def extract_text_from_hybrid_pdf(self) -> List[Document]:
        """Extract text from a hybrid PDF (text + images) using UnstructuredPDFLoader.
        
        This method uses UnstructuredPDFLoader which can handle PDFs with both
        text and image content, though it requires additional dependencies.
        
        Returns:
            List of Document objects containing chunked text
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            RuntimeError: If extraction fails or dependencies missing
        """
        path = Path(self.pdf_path)
        if not path.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        try:
            from langchain_community.document_loaders import UnstructuredPDFLoader
            
            logger.info(f"Loading hybrid PDF: {self.pdf_path}")
            loader = UnstructuredPDFLoader(str(path))
            pages = loader.load()
            logger.info(f"Loaded {len(pages)} pages from hybrid PDF")
            
        except ImportError as e:
            logger.error("UnstructuredPDFLoader not available. Install: pip install unstructured")
            raise RuntimeError(
                "UnstructuredPDFLoader not available. "
                "Install required packages: pip install unstructured"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load hybrid PDF {self.pdf_path}: {e}")
            raise RuntimeError(f"Hybrid PDF extraction failed: {e}") from e
        
        # Split pages into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(pages)
        logger.info(f"✅ Split {self.pdf_path} into {len(chunks)} text chunks")

        return chunks

