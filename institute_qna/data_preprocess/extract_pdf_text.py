"""PDF Text Extraction Module.

This module provides utilities to extract text from PDF files and split them
into manageable chunks for processing and embedding generation.

Supports two extraction methods:
1. 'opensource' - Uses pypdf + pdfplumber (default, no API costs)
2. 'azure' - Uses Azure Document Intelligence (requires Azure credentials)
"""

from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional, Tuple, Literal
import csv
import hashlib
import logging
import re
import os
import io
import random
from datetime import datetime
from time import sleep
import requests

logger = logging.getLogger(__name__)

# Try importing pdfplumber for better table extraction
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    logger.warning("pdfplumber not available. Table extraction will be limited. Install with: pip install pdfplumber")

# Try importing Azure Document Intelligence
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.core.exceptions import HttpResponseError, ServiceRequestError, ServiceResponseError
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
    AZURE_DOC_INTELLIGENCE_AVAILABLE = True
except ImportError:
    AZURE_DOC_INTELLIGENCE_AVAILABLE = False
    logger.info("Azure Document Intelligence not available. Install with: pip install azure-ai-documentintelligence")

try:
    from azure.storage.blob import BlobServiceClient
    AZURE_BLOB_AVAILABLE = True
except ImportError:
    AZURE_BLOB_AVAILABLE = False


class PDFTextExtractor:
    """Utility class to extract text from PDF files and split into chunks.
    
    This class handles both text-based and hybrid PDFs, providing methods to
    extract and chunk text content for downstream processing.
    """

    def __init__(
                self, 
                pdf_path: str = "attachments/brochure.pdf",
                chunk_size: int = 2000, 
                chunk_overlap: int = 200,
                tables_output_dir: str = "extracted_text_data/checkpoints/tables",
                university: str = "coep",
                skip_merit_lists: bool = True,
                extraction_method: Literal["opensource", "azure"] = "azure"
            ):
        """Initialize the PDF Text Extractor.
        
        Args:
            pdf_path: Path to the PDF file to extract
            chunk_size: Size of text chunks (in characters)
            chunk_overlap: Overlap between chunks (in characters)
            tables_output_dir: Directory to save extracted tables as CSV
            university: University identifier for metadata
            skip_merit_lists: Whether to skip merit list PDFs
            extraction_method: Method to use - 'opensource' (pypdf+pdfplumber) or 'azure' (Azure Document Intelligence)
        """
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tables_output_dir = Path(tables_output_dir)
        self.university = university
        self.skip_merit_lists = skip_merit_lists
        self.extraction_method = extraction_method
        self.extracted_text: List[Document] = []
        self._table_schema_paths: Dict[str, str] = {}
        self._blob_service_client = None
        self._blob_container_client = None
        self.azure_blob_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.azure_blob_container_name = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
        self.tables_blob_prefix = os.getenv(
            "AZURE_TABLES_BLOB_PREFIX",
            f"tables/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Validate extraction method and dependencies
        if self.extraction_method == "azure" and not AZURE_DOC_INTELLIGENCE_AVAILABLE:
            logger.error("Azure Document Intelligence not available. Install with: pip install azure-ai-documentintelligence")
            raise ImportError("Azure Document Intelligence SDK not installed")
        
        # Initialize Azure client if using Azure method
        self.azure_client = None
        if self.extraction_method == "azure":
            self._initialize_azure_client()
            self._initialize_blob_client()

    def _initialize_blob_client(self) -> None:
        """Initialize Azure Blob container client for table CSV uploads."""
        if not AZURE_BLOB_AVAILABLE:
            raise ImportError(
                "azure-storage-blob is not installed. "
                "Install with: pip install azure-storage-blob"
            )
        if not self.azure_blob_connection_string:
            raise ValueError(
                "AZURE_STORAGE_CONNECTION_STRING is required for blob-backed table exports"
            )

        self._blob_service_client = BlobServiceClient.from_connection_string(
            self.azure_blob_connection_string
        )
        self._blob_container_client = self._blob_service_client.get_container_client(
            self.azure_blob_container_name
        )
        if not self._blob_container_client.exists():
            self._blob_container_client.create_container()

    def _upload_csv_to_blob(self, blob_name: str, rows: List[List[str]]) -> str:
        """Upload CSV rows to blob and return blob path string."""
        if not self._blob_container_client:
            raise RuntimeError("Blob container client is not initialized")

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        for row in rows:
            clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
            writer.writerow(clean_row)

        blob_path = f"{self.tables_blob_prefix}/{blob_name}"
        blob_client = self._blob_container_client.get_blob_client(blob=blob_path)
        blob_client.upload_blob(buffer.getvalue().encode("utf-8"), overwrite=True)
        return f"azure-blob://{self.azure_blob_container_name}/{blob_path}"
    
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
                                'raw_table': table,
                                'header': table[0] if table else [],
                                'rows': len(table),
                                'cols': len(table[0]) if table else 0
                            })
                            logger.info(f"Extracted table from page {page_num + 1}: {len(table)} rows")
        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
        
        return tables_data

    def get_table_schema_key(self, header: List[str]) -> str:
        """Return a stable schema key for table header."""
        normalized = [str(cell).strip().lower() for cell in header if cell is not None]
        header_joined = "|".join(normalized)
        return hashlib.md5(header_joined.encode("utf-8")).hexdigest() if header_joined else "no_header"

    def write_table_to_csv(
        self,
        table: List[List[str]],
        pdf_stem: str,
        page_num: int,
        table_index: int,
        schema_key: str
    ) -> str:
        """Write a table CSV to Azure Blob and return the blob URI."""
        schema_part = schema_key[:12] if schema_key else "no_header"
        filename = f"{pdf_stem}_p{page_num}_t{table_index}_{schema_part}.csv"
        blob_uri = self._upload_csv_to_blob(filename, table)
        self._table_schema_paths[schema_key] = blob_uri
        return blob_uri

    def write_merit_table_to_csv(
        self,
        table: List[List[str]],
        pdf_stem: str,
        page_num: int,
        table_index: int
    ) -> str:
        """Write merit list table CSV to Azure Blob and return the blob URI."""
        filename = f"{pdf_stem}_p{page_num}_t{table_index}_merit.csv"
        return self._upload_csv_to_blob(filename, table)
    
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

    def is_index_page(self, text: str) -> bool:
        """Heuristic check for index/table-of-contents pages."""
        if not text:
            return False
        lowered = text.lower()
        if "table of contents" in lowered or "contents" in lowered or "index" in lowered:
            dot_leader_lines = re.findall(r"\.\.{2,}\s*\d+", text)
            if len(dot_leader_lines) >= 2:
                return True

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return False
        dot_leader_ratio = sum(1 for line in lines if re.search(r"\.\.{2,}\s*\d+", line)) / len(lines)
        if dot_leader_ratio >= 0.3 and len(lines) >= 6:
            return True
        return False
    
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
                 for term in ['merit', 'merit list', 'rank list']):
            return 'merit_list'
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
        
        This method routes to the appropriate extraction method based on 
        the extraction_method parameter:
        - 'opensource': Uses PyPDFLoader + pdfplumber (default)
        - 'azure': Uses Azure Document Intelligence
        
        Returns:
            List of Document objects containing chunked text
            
        Raises:
            FileNotFoundError: If PDF file does not exist
            RuntimeError: If PDF extraction fails
        """
        # Route to appropriate extraction method
        if self.extraction_method == "azure":
            return self.extract_text_from_azure_doc_intelligence()
        else:
            return self._extract_text_opensource()
    
    def _extract_text_opensource(self) -> List[Document]:
        """Extract text from PDF using open-source tools (PyPDF + pdfplumber).
        
        Returns:
            List of Document objects containing chunked text
            
        Raises:
            FileNotFoundError: If PDF file does not exist
            RuntimeError: If PDF extraction fails
        """
        path = Path(self.pdf_path)
        if not path.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")

        try:
            logger.info(f"Loading PDF: {self.pdf_path}")
            loader = PyPDFLoader(str(path),extract_images=True)
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
        if "merit" in path.name.lower():
            doc_type = "merit_list"

        if doc_type == "merit_list" and self.skip_merit_lists:
            logger.info(f"Skipping merit list PDF: {path.name}")
            return []
        
        # Extract metadata from full text
        full_text = " ".join([p.page_content for p in pages[:3]])  # First 3 pages for metadata
        extracted_metadata = self.extract_metadata_from_pdf_text(full_text)
        
        table_documents: List[Document] = []
        enhanced_pages = []
        plumber_pdf = None
        if PDFPLUMBER_AVAILABLE:
            try:
                plumber_pdf = pdfplumber.open(str(path))
            except Exception as e:
                logger.warning(f"pdfplumber text fallback not available for {path.name}: {e}")

        skipped_index_pages = 0
        for idx, page in enumerate(pages):
            page_text = page.page_content or ""

            if plumber_pdf is not None and len(page_text.strip()) < 50:
                try:
                    fallback_text = plumber_pdf.pages[idx].extract_text() or ""
                    if len(fallback_text.strip()) > len(page_text.strip()):
                        page_text = fallback_text
                        page.page_content = page_text
                except Exception:
                    pass

            if self.is_index_page(page_text):
                skipped_index_pages += 1
                continue

            page_tables = [t for t in tables_data if t['page'] == idx + 1]
            table_csv_paths = []
            for table_info in page_tables:
                if doc_type == "merit_list":
                    csv_path = self.write_merit_table_to_csv(
                        table_info['raw_table'],
                        path.stem,
                        idx + 1,
                        table_info['table_index']
                    )
                    schema_key = "merit_list"
                else:
                    schema_key = self.get_table_schema_key(table_info.get("header", []))
                    csv_path = self.write_table_to_csv(
                        table_info['raw_table'],
                        path.stem,
                        idx + 1,
                        table_info['table_index'],
                        schema_key
                    )
                table_csv_paths.append(csv_path)
                table_info['csv_path'] = csv_path
                table_info['schema_key'] = schema_key

                table_documents.append(
                    Document(
                        page_content=table_info['content'],
                        metadata={
                            "source": str(path),
                            "filename": path.name,
                            "doc_type": doc_type,
                            "page": idx + 1,
                            "table_index": table_info['table_index'],
                            "table_csv_path": csv_path,
                            "table_schema_key": schema_key,
                            "table_schema_header": table_info.get("header", []),
                            "has_table": True,
                            "source_type": "pdf_table",
                            "university": self.university,
                            "extraction_method": "opensource",
                            **extracted_metadata
                        }
                    )
                )

            page_has_table = self.detect_tables_in_text(page_text) or bool(page_tables)

            page.metadata.update({
                "source": page.metadata.get("source", str(path)),
                "doc_type": doc_type,
                "filename": path.name,
                "page_number": idx + 1,
                "has_table": page_has_table,
                "table_csv_paths": table_csv_paths,
                "source_type": "pdf",
                "university": self.university,
                "extraction_method": "opensource",
                **extracted_metadata
            })
            enhanced_pages.append(page)

        if plumber_pdf is not None:
            try:
                plumber_pdf.close()
            except Exception:
                pass

        if skipped_index_pages:
            logger.info(f"Skipped {skipped_index_pages} index/contents page(s) in {path.name}")
        
        # Use smart separators based on content type
        separators = ["\n\n", "\n", ". ", " ", ""]
        if doc_type in ['fees', 'admissions']:
            # For structured documents, respect list items and table rows
            separators = ["\n### ", "\n## ", "\n\n", "\n", ". ", " ", ""]
        
        if doc_type == "merit_list":
            # Merit list documents are table-only; keep CSVs and table docs only.
            self.extracted_text = table_documents
        else:
            # Split pages into smaller chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=separators
            )

            self.extracted_text = splitter.split_documents(enhanced_pages)
            if table_documents:
                self.extracted_text.extend(table_documents)
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
            FileNotFoundError: If folder does not exist
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
      
    def _initialize_azure_client(self) -> None:
        """Initialize Azure Document Intelligence client from environment variables.
        
        Requires environment variables:
        - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
        - AZURE_DOCUMENT_INTELLIGENCE_KEY
        
        Raises:
            ValueError: If required environment variables are not set
        """
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        
        if not endpoint or not key:
            logger.error("Azure Document Intelligence credentials not found in environment variables")
            raise ValueError(
                "Missing Azure credentials. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT "
                "and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables"
            )
        
        self.azure_client = DocumentIntelligenceClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )
        logger.info("Azure Document Intelligence client initialized")

    def _get_retry_after_seconds(self, error: Exception) -> Optional[float]:
        """Parse retry-after hint from Azure error response headers when available."""
        response = getattr(error, "response", None)
        if not response:
            return None

        headers = getattr(response, "headers", {}) or {}
        raw_retry_after = (
            headers.get("Retry-After")
            or headers.get("retry-after")
            or headers.get("x-ms-retry-after-ms")
            or headers.get("x-ms-retry-after")
        )
        if raw_retry_after is None:
            return None

        try:
            value = float(raw_retry_after)
        except (TypeError, ValueError):
            return None

        if "ms" in str(raw_retry_after).lower() or "x-ms-retry-after-ms" in headers:
            return max(0.0, value / 1000.0)
        return max(0.0, value)

    def _is_retryable_azure_error(self, error: Exception) -> bool:
        """Return True for transient/rate-limit errors that should be retried."""
        if isinstance(error, (ServiceRequestError, ServiceResponseError, TimeoutError, requests.Timeout)):
            return True

        if isinstance(error, HttpResponseError):
            status_code = getattr(error, "status_code", None)
            if status_code in {408, 409, 429, 500, 502, 503, 504}:
                return True

            response = getattr(error, "response", None)
            if response is not None:
                try:
                    status_code = response.status_code
                except Exception:
                    pass
                if status_code in {408, 409, 429, 500, 502, 503, 504}:
                    return True

        return False

    def _analyze_pdf_with_retry(self, pdf_bytes: bytes, source_name: str):
        """Call Azure DI with retry/backoff tuned for free-tier throttling."""
        max_attempts = max(1, int(os.getenv("AZURE_DOC_INTEL_MAX_RETRIES", "6")))
        base_delay_seconds = max(1.0, float(os.getenv("AZURE_DOC_INTEL_BASE_DELAY_SECONDS", "4")))
        max_delay_seconds = max(base_delay_seconds, float(os.getenv("AZURE_DOC_INTEL_MAX_DELAY_SECONDS", "60")))

        for attempt in range(1, max_attempts + 1):
            try:
                poller = self.azure_client.begin_analyze_document(
                    model_id="prebuilt-layout",
                    body=pdf_bytes,
                    content_type="application/pdf",
                    output_content_format=DocumentContentFormat.MARKDOWN,
                )
                return poller.result()
            except Exception as error:
                if attempt >= max_attempts or not self._is_retryable_azure_error(error):
                    raise

                retry_after_seconds = self._get_retry_after_seconds(error)
                exponential_delay = min(max_delay_seconds, base_delay_seconds * (2 ** (attempt - 1)))
                jitter = random.uniform(0.0, 1.5)
                delay_seconds = max(retry_after_seconds or 0.0, exponential_delay + jitter)

                logger.warning(
                    "Azure DI transient failure for %s (attempt %d/%d): %s. Retrying in %.1f sec",
                    source_name,
                    attempt,
                    max_attempts,
                    error,
                    delay_seconds,
                )
                sleep(delay_seconds)
    
    def extract_text_from_azure_doc_intelligence(self) -> List[Document]:
        """Extract text and tables from PDF using Azure Document Intelligence.
        
        This method uses Azure's Document Intelligence service to analyze PDFs,
        providing high-quality extraction of text, tables, and document structure.
        
        Returns:
            List of Document objects containing chunked text with table information
            
        Raises:
            FileNotFoundError: If PDF file does not exist
            RuntimeError: If Azure extraction fails
        """
        path = Path(self.pdf_path)
        if not path.exists():
            logger.error(f"PDF file not found: {self.pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        if not self.azure_client:
            logger.error("Azure client not initialized")
            raise RuntimeError("Azure Document Intelligence client not initialized")
        
        logger.info(f"Analyzing PDF with Azure Document Intelligence: {self.pdf_path}")

        try:
            with open(path, "rb") as f:
                pdf_bytes = f.read()
            return self._extract_text_from_azure_pdf_bytes(
                pdf_bytes=pdf_bytes,
                source_name=path.name,
                source_reference=str(path),
            )

        except Exception as e:
            logger.error(f"Azure Document Intelligence extraction failed for {self.pdf_path}: {e}")
            raise RuntimeError(f"Azure extraction failed: {e}") from e

    def _extract_text_from_azure_pdf_bytes(
        self,
        pdf_bytes: bytes,
        source_name: str,
        source_reference: str,
    ) -> List[Document]:
        """Extract text from PDF bytes using Azure Document Intelligence."""
        try:
            result = self._analyze_pdf_with_retry(
                pdf_bytes=pdf_bytes,
                source_name=source_name,
            )
            logger.info(f"Azure analysis complete for {source_name}")
            source_stem = Path(source_name).stem
            
            # Classify document type
            sample_text = result.content[:1000] if result.content else ""
            doc_type = self.classify_pdf_document(sample_text, source_name)
            
            if "merit" in source_name.lower():
                doc_type = "merit_list"
            
            if doc_type == "merit_list" and self.skip_merit_lists:
                logger.info(f"Skipping merit list PDF: {source_name}")
                return []
            
            # Extract metadata from content
            extracted_metadata = self.extract_metadata_from_pdf_text(sample_text)
            
            # Process tables if present
            table_documents: List[Document] = []
            table_info_by_page: Dict[int, List[Dict]] = {}
            
            if result.tables:
                logger.info(f"Found {len(result.tables)} tables in {source_name}")
                for table_idx, table in enumerate(result.tables):
                    # Convert Azure table to list format
                    table_data = self._convert_azure_table_to_list(table)
                    
                    # Determine page number (Azure uses 1-based page numbers)
                    page_num = self._get_table_page_number(table)
                    
                    # Format table as markdown
                    formatted_table = self.format_table_as_markdown(table_data)
                    
                    # Determine if it's a merit list table
                    header = table_data[0] if table_data else []
                    is_merit_table = self._is_merit_list_table(header)
                    
                    # Save table to CSV
                    if is_merit_table and doc_type == "merit_list":
                        csv_path = self.write_merit_table_to_csv(
                            table_data,
                            source_stem,
                            page_num,
                            table_idx
                        )
                        schema_key = "merit_list"
                    else:
                        schema_key = self.get_table_schema_key(header)
                        csv_path = self.write_table_to_csv(
                            table_data,
                            source_stem,
                            page_num,
                            table_idx,
                            schema_key
                        )
                    
                    # Store table info
                    if page_num not in table_info_by_page:
                        table_info_by_page[page_num] = []
                    
                    table_info_by_page[page_num].append({
                        'table_index': table_idx,
                        'content': formatted_table,
                        'csv_path': csv_path,
                        'schema_key': schema_key,
                        'header': header
                    })
                    
                    # Create document for table
                    table_documents.append(
                        Document(
                            page_content=formatted_table,
                            metadata={
                                "source": source_reference,
                                "filename": source_name,
                                "doc_type": doc_type,
                                "page": page_num,
                                "table_index": table_idx,
                                "table_csv_path": csv_path,
                                "table_schema_key": schema_key,
                                "table_schema_header": header,
                                "has_table": True,
                                "source_type": "pdf_table",
                                "university": self.university,
                                "extraction_method": "azure",
                                **extracted_metadata
                            }
                        )
                    )
            
            # Process main content as pages
            # Split content by pages if page information is available
            pages_content = []
            if result.pages:
                for page_idx, page in enumerate(result.pages):
                    page_num = page_idx + 1
                    
                    # Extract text from this page using spans
                    page_text = self._extract_page_text_from_result(result, page)
                    
                    if self.is_index_page(page_text):
                        logger.debug(f"Skipping index page {page_num}")
                        continue
                    
                    # Check if page has tables
                    page_tables = table_info_by_page.get(page_num, [])
                    has_table = len(page_tables) > 0 or self.detect_tables_in_text(page_text)
                    table_csv_paths = [t['csv_path'] for t in page_tables]
                    
                    # Create page document
                    page_doc = Document(
                        page_content=page_text,
                        metadata={
                            "source": source_reference,
                            "filename": source_name,
                            "doc_type": doc_type,
                            "page_number": page_num,
                            "has_table": has_table,
                            "table_csv_paths": table_csv_paths,
                            "source_type": "pdf",
                            "university": self.university,
                            "extraction_method": "azure",
                            **extracted_metadata
                        }
                    )
                    pages_content.append(page_doc)
            else:
                # Fallback: use entire content as single document
                logger.warning("No page information available, using full content")
                page_doc = Document(
                    page_content=result.content,
                    metadata={
                            "source": source_reference,
                            "filename": source_name,
                        "doc_type": doc_type,
                        "source_type": "pdf",
                        "university": self.university,
                        "extraction_method": "azure",
                        **extracted_metadata
                    }
                )
                pages_content.append(page_doc)
            
            # Smart separators based on content type
            separators = ["\n\n", "\n", ". ", " ", ""]
            if doc_type in ['fees', 'admissions']:
                separators = ["\n### ", "\n## ", "\n\n", "\n", ". ", " ", ""]
            
            # Handle merit list documents
            if doc_type == "merit_list":
                self.extracted_text = table_documents
            else:
                # Split pages into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    length_function=len,
                    separators=separators
                )
                
                self.extracted_text = splitter.split_documents(pages_content)
                
                # Add table documents
                if table_documents:
                    self.extracted_text.extend(table_documents)
            
            logger.info(
                f"✅ Extracted from {source_name} using Azure: {len(self.extracted_text)} chunks "
                f"(Type: {doc_type}, Tables: {len(result.tables) if result.tables else 0})"
            )
            
            return self.extracted_text
            
        except Exception as e:
            logger.error(f"Azure Document Intelligence extraction failed for {source_name}: {e}")
            raise RuntimeError(f"Azure extraction failed: {e}") from e

    def extract_multiple_pdfs_from_urls(self, pdf_urls: List[str]) -> List[Document]:
        """Extract PDF text directly from URLs without writing files locally.

        This method is Azure-only because open-source extraction expects filesystem paths.
        """
        if self.extraction_method != "azure":
            raise ValueError("URL-based extraction without local files is only supported for extraction_method='azure'")

        if not pdf_urls:
            logger.warning("No PDF URLs provided for extraction")
            return []

        all_chunks: List[Document] = []
        successful = 0
        failed = 0
        seen = set()

        for url in pdf_urls:
            if not url or url in seen:
                continue
            seen.add(url)

            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                filename = Path(url.split("?")[0]).name or "document.pdf"
                chunks = self._extract_text_from_azure_pdf_bytes(
                    pdf_bytes=response.content,
                    source_name=filename,
                    source_reference=url,
                )
                all_chunks.extend(chunks)
                successful += 1
            except Exception as e:
                logger.error("Failed to process PDF URL %s: %s", url, e)
                failed += 1

        logger.info(
            "Completed URL PDF extraction: %d successful, %d failed, %d total chunks",
            successful,
            failed,
            len(all_chunks),
        )
        return all_chunks
    
    def _convert_azure_table_to_list(self, table) -> List[List[str]]:
        """Convert Azure Document Intelligence table to list of lists format.
        
        Args:
            table: Azure table object
            
        Returns:
            List of lists representing the table
        """
        if not table.cells:
            return []
        
        # Find table dimensions
        max_row = max(cell.row_index for cell in table.cells) + 1
        max_col = max(cell.column_index for cell in table.cells) + 1
        
        # Initialize empty table
        table_data = [["" for _ in range(max_col)] for _ in range(max_row)]
        
        # Fill in cell values
        for cell in table.cells:
            row = cell.row_index
            col = cell.column_index
            content = cell.content or ""
            
            # Handle cell spanning
            row_span = getattr(cell, 'row_span', 1) or 1
            col_span = getattr(cell, 'column_span', 1) or 1
            
            # Fill all spanned cells
            for r in range(row, min(row + row_span, max_row)):
                for c in range(col, min(col + col_span, max_col)):
                    if r == row and c == col:
                        table_data[r][c] = content
                    else:
                        table_data[r][c] = ""  # Mark spanned cells as empty
        
        return table_data
    
    def _get_table_page_number(self, table) -> int:
        """Extract page number from Azure table object.
        
        Args:
            table: Azure table object
            
        Returns:
            Page number (1-indexed)
        """
        # Try to get from bounding regions
        if hasattr(table, 'bounding_regions') and table.bounding_regions:
            return table.bounding_regions[0].page_number
        
        # Fallback to 1 if not available
        return 1
    
    def _extract_page_text_from_result(self, result, page) -> str:
        """Extract text content for a specific page from Azure result.
        
        Args:
            result: Azure analysis result
            page: Page object
            
        Returns:
            Text content for the page
        """
        # Use page spans to extract relevant text from full content
        page_text = []
        
        if hasattr(page, 'spans') and page.spans:
            for span in page.spans:
                offset = span.offset
                length = span.length
                page_text.append(result.content[offset:offset + length])
        
        return "\n".join(page_text) if page_text else ""
    
    def _is_merit_list_table(self, header: List[str]) -> bool:
        """Check if table header indicates merit list.
        
        Args:
            header: Table header row
            
        Returns:
            True if this is a merit list table
        """
        header_text = " ".join(str(h).lower() for h in header)
        merit_indicators = ['rank', 'merit', 'score', 'percentile', 'marks', 'candidate']
        return sum(indicator in header_text for indicator in merit_indicators) >= 2

