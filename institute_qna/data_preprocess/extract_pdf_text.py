"""PDF Text Extraction Module.

This module provides utilities to extract text from PDF files and split them
into manageable chunks for processing and embedding generation.
"""

from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

# Try importing pdfplumber for better table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Table extraction will be limited. Install with: pip install pdfplumber")


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
    
    def detect_tables_in_text(self, text: str) -> bool:
        """Detect if text contains table-like structures.
        
        Args:
            text: Text content to analyze
            
        Returns:
            True if table-like patterns detected
        """
        # Look for common table patterns
        table_indicators = [
            r'\|.*\|.*\|',  # Pipe-separated columns
            r'[-─]{3,}',  # Horizontal lines
            r'\s{3,}\w+\s{3,}\w+',  # Multiple spaces between words (columns)
            r'(?:\d+\.?\s*){3,}',  # Series of numbers (often in tables)
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text):
                return True
        return False
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract tables from PDF using pdfplumber.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing table data and metadata
        """
        if not PDFPLUMBER_AVAILABLE:
            logger.warning("pdfplumber not available, skipping table extraction")
            return []
        
        tables_data = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    for table_idx, table in enumerate(tables):
                        if table and len(table) > 1:  # At least header + 1 row
                            # Format table as markdown
                            formatted_table = self.format_table_as_markdown(table)
                            tables_data.append({
                                'page': page_num + 1,
                                'table_index': table_idx,
                                'content': formatted_table,
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0
                            })
                            logger.info(f"Extracted table from page {page_num + 1}: {len(table)} rows")
        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
        
        return tables_data
    
    def format_table_as_markdown(self, table: List[List[str]]) -> str:
        """Format table data as markdown for better chunking.
        
        Args:
            table: 2D list representing table data
            
        Returns:
            Markdown-formatted table string
        """
        if not table:
            return ""
        
        # Clean None values and empty cells
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) < 2:
            return "\n".join([" | ".join(row) for row in cleaned_table])
        
        # Create markdown table
        markdown_lines = []
        
        # Header
        header = " | ".join(cleaned_table[0])
        markdown_lines.append(header)
        
        # Separator
        separator = " | ".join(["---"] * len(cleaned_table[0]))
        markdown_lines.append(separator)
        
        # Data rows
        for row in cleaned_table[1:]:
            markdown_lines.append(" | ".join(row))
        
        return "\n".join(markdown_lines)
    
    def classify_pdf_document(self, text_sample: str, filename: str) -> str:
        """Classify PDF document type based on content and filename.
        
        Args:
            text_sample: Sample text from PDF
            filename: PDF filename
            
        Returns:
            Document type classification
        """
        text_lower = text_sample.lower()
        filename_lower = filename.lower()
        
        if any(term in filename_lower or term in text_lower 
               for term in ['fee', 'fees', 'payment', 'tuition', 'cost']):
            return 'fees'
        elif any(term in filename_lower or term in text_lower 
                 for term in ['brochure', 'prospectus', 'overview']):
            return 'brochure'
        elif any(term in filename_lower or term in text_lower 
                 for term in ['admission', 'eligibility', 'entrance']):
            return 'admissions'
        elif any(term in filename_lower or term in text_lower 
                 for term in ['manual', 'guide', 'instruction', 'how to']):
            return 'manual'
        elif any(term in filename_lower or term in text_lower 
                 for term in ['application', 'form', 'process']):
            return 'application'
        else:
            return 'general'
    
    def extract_metadata_from_pdf_text(self, text: str) -> Dict:
        """Extract structured metadata from PDF text.
        
        Args:
            text: PDF text content
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract dates
        dates = re.findall(
            r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
            text, re.IGNORECASE
        )
        if dates:
            metadata['dates'] = list(set(dates[:5]))
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata['emails'] = list(set(emails[:3]))
        
        # Extract phone numbers (Indian format)
        phones = re.findall(r'\b(?:\+91[\s-]?)?\d{3}[\s-]?\d{3}[\s-]?\d{4}\b', text)
        if phones:
            metadata['phones'] = list(set(phones[:3]))
        
        # Extract academic years
        academic_years = re.findall(r'\b(?:AY|Academic Year)[\s:]*(\d{4}[-/]\d{2,4})\b', text, re.IGNORECASE)
        if academic_years:
            metadata['academic_year'] = academic_years[0]
        
        return metadata

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
        
        # Extract tables if pdfplumber is available
        tables_data = self.extract_tables_from_pdf(str(path))
        
        # Classify document type
        sample_text = pages[0].page_content[:1000] if pages else ""
        doc_type = self.classify_pdf_document(sample_text, path.name)
        
        # Extract metadata from full text
        full_text = " ".join([p.page_content for p in pages[:3]])  # First 3 pages for metadata
        extracted_metadata = self.extract_metadata_from_pdf_text(full_text)
        
        # Enhance page metadata
        enhanced_pages = []
        for idx, page in enumerate(pages):
            # Check if page has tables
            page_has_table = self.detect_tables_in_text(page.page_content)
            
            # Add table content if found on this page
            page_tables = [t['content'] for t in tables_data if t['page'] == idx + 1]
            if page_tables:
                # Append formatted tables to page content
                table_section = "\n\n### Tables on this page:\n\n" + "\n\n".join(page_tables)
                page.page_content += table_section
            
            # Enhance metadata
            page.metadata.update({
                'doc_type': doc_type,
                'filename': path.name,
                'has_table': page_has_table or bool(page_tables),
                **extracted_metadata
            })
            enhanced_pages.append(page)
        
        # Use smart separators based on content type
        separators = ["\n\n", "\n", ". ", " ", ""]
        if doc_type in ['fees', 'admissions']:
            # For structured documents, respect list items and table rows
            separators = ["\n### ", "\n## ", "\n\n", "\n", ". ", " ", ""]
        
        # Split pages into smaller chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=separators
        )

        self.extracted_text = splitter.split_documents(enhanced_pages)
        logger.info(
            f"✅ Split {self.pdf_path} into {len(self.extracted_text)} chunks "
            f"(Type: {doc_type}, Tables: {len(tables_data)})"
        )

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
        doc_types_count = {}
        tables_count = 0
        
        for pdf_path in pdf_paths:
            try:
                self.pdf_path = str(pdf_path)
                chunks = self.extract_text_from_text_pdf()
                
                # Track document types
                if chunks:
                    doc_type = chunks[0].metadata.get('doc_type', 'unknown')
                    doc_types_count[doc_type] = doc_types_count.get(doc_type, 0) + 1
                    
                    # Count tables
                    tables_in_doc = sum(1 for c in chunks if c.metadata.get('has_table', False))
                    tables_count += tables_in_doc
                
                all_chunks.extend(chunks)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                failed += 1
                continue
        
        logger.info(
            f"Completed PDF extraction: {successful} successful, "
            f"{failed} failed, {len(all_chunks)} total chunks, "
            f"{tables_count} chunks with tables"
        )
        logger.info(f"Document types: {doc_types_count}")
        
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

