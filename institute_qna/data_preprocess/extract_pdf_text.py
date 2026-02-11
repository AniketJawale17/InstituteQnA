"""PDF Text Extraction Module.

This module provides utilities to extract text from PDF files and split them
into manageable chunks for processing and embedding generation.
"""

from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Dict, Optional, Tuple
import csv
import hashlib
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
                            chunk_overlap: int = 200,
                              tables_output_dir: str = "extracted_text_data/checkpoints/tables",
                            university: str = "coep",
                            skip_merit_lists: bool = True
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
        self.tables_output_dir = Path(tables_output_dir)
        self.university = university
        self.skip_merit_lists = skip_merit_lists
        self.extracted_text: List[Document] = []
        self._table_schema_paths: Dict[str, str] = {}
    
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
        """Write a table to CSV and return the relative file path."""
        self.tables_output_dir.mkdir(parents=True, exist_ok=True)
        if schema_key == "no_header":
            filename = f"{pdf_stem}_table_{table_index}_no_header.csv"
        else:
            filename = f"table_schema_{schema_key}.csv"
        output_path = self.tables_output_dir / filename

        write_header = True
        if schema_key in self._table_schema_paths and output_path.exists():
            write_header = False

        mode = "a" if output_path.exists() else "w"
        with open(output_path, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for row_idx, row in enumerate(table):
                clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                if row_idx == 0 and not write_header:
                    continue
                writer.writerow(clean_row)

        self._table_schema_paths[schema_key] = str(output_path.as_posix())

        return str(output_path.as_posix())

    def write_merit_table_to_csv(
        self,
        table: List[List[str]],
        pdf_stem: str,
        page_num: int,
        table_index: int
    ) -> str:
        """Append merit list tables into a single CSV without schema checks."""
        self.tables_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.tables_output_dir / "merit_list_tables.csv"

        mode = "a" if output_path.exists() else "w"
        with open(output_path, mode, newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            for row in table:
                clean_row = [str(cell).strip() if cell is not None else "" for cell in row]
                writer.writerow(clean_row)

        return str(output_path.as_posix())
    
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

        filtered_pages = []
        skipped_index_pages = 0
        for idx, page in enumerate(pages):
            if self.is_index_page(page.page_content):
                skipped_index_pages += 1
                continue
            page.metadata.update({
                "source": page.metadata.get("source", str(path)),
                "filename": path.name,
                "page_number": idx + 1,
                "source_type": "pdf",
                "university": self.university
            })
            filtered_pages.append(page)

        if skipped_index_pages:
            logger.info(f"Skipped {skipped_index_pages} index/contents page(s) in {path.name}")

        chunks = splitter.split_documents(filtered_pages)
        logger.info(f"✅ Split {self.pdf_path} into {len(chunks)} text chunks")

        return chunks

