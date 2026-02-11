"""Knowledge Base Creation Module.

This module structures extracted text from PDFs and websites into a format 
suitable for knowledge base creation and vector embedding.

Complete Pipeline:
1. Web scraping (data extraction)
2. Attachment downloading (PDFs, docs)
3. Document processing and cleaning
4. Chunking and deduplication
5. Checkpointing at each step
6. Embedding generation
"""

from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
from institute_qna.data_extraction.webscrapper import WebBasedLoader
from institute_qna.data_extraction.download_attachment import AttachmentDownloader
import json
import re
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
from institute_qna.data_preprocess.embedding_generation import EmbeddingsGeneration
from langchain_core.documents import Document 
from pathlib import Path
from typing import List, Set, Optional
import logging
from time import sleep
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class KnowledgeBaseCreation(PDFTextExtractor):
    """Generates Knowledge base through structuring extracted text.
    
    This class processes both PDF documents and web-scraped data, structuring
    them into a unified format suitable for embedding and retrieval.
    
    Includes checkpoint saving at each step to prevent data loss.
    """
    
    # Common UI patterns to remove from markdown content
    NOISE_PATTERNS = [
        r'\[.*?Login.*?\]\(.*?\)',  # Login links
        r'Accessibility Tools',
        r'\* Invert colors.*?Letter spacing\s+100%',  # Accessibility menu
        r'Search\s+Search',
        r'\[-A\].*?\[\+A\]',  # Font size controls
        r'Menu\n',
        r'Skip to content',
        r'Copyright.*?All rights reserved.*',
        r'Best Viewed in.*?\d+ x \d+.*',
        r'\d+\n\d+\n\d+\n\d+ Visitors',  # Visitor counter
    ]
    
    def __init__(
        self,
        checkpoint_dir: str = "extracted_text_data/checkpoints",
        university: str = "coep"
    ):
        """Initialize the Knowledge Base Creation class.
        
        Args:
            checkpoint_dir: Directory to save checkpoint files
        """
        super().__init__(university=university)
        self.structured_documents: List[Document] = []
        self._seen_content_hashes: Set[str] = set()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.university = university
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def save_checkpoint(self, data: any, step_name: str, timestamp: Optional[str] = None) -> Path:
        """Save checkpoint data to prevent data loss.
        
        Args:
            data: Data to save (list, dict, etc.)
            step_name: Name of the processing step
            timestamp: Optional timestamp string, defaults to current time
            
        Returns:
            Path to saved checkpoint file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_file = self.checkpoint_dir / f"{step_name}_{timestamp}.json"
        
        try:
            # Convert Documents to serializable format if needed
            if isinstance(data, list) and data and isinstance(data[0], Document):
                serializable_data = [
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in data
                ]
            else:
                serializable_data = data
            
            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Checkpoint saved: {checkpoint_file} ({len(data) if isinstance(data, list) else 'N/A'} items)")
            return checkpoint_file
        except Exception as e:
            logger.error(f"Failed to save checkpoint {step_name}: {e}")
            raise

    def clear_tables_checkpoint_dir(self) -> None:
        """Remove existing table CSVs to avoid duplicates between runs."""
        tables_dir = Path(self.tables_output_dir)
        if not tables_dir.exists():
            return
        for entry in tables_dir.iterdir():
            if entry.is_file():
                entry.unlink()
    
    def load_checkpoint(self, checkpoint_file: Path, as_documents: bool = False) -> any:
        """Load data from checkpoint file.
        
        Args:
            checkpoint_file: Path to checkpoint file
            as_documents: If True, convert data back to Document objects
            
        Returns:
            Loaded data
        """
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if as_documents and isinstance(data, list):
                documents = [
                    Document(
                        page_content=item.get('page_content', ''),
                        metadata=item.get('metadata', {})
                    )
                    for item in data
                ]
                logger.info(f"Loaded {len(documents)} documents from checkpoint")
                return documents
            
            logger.info(f"Loaded checkpoint from {checkpoint_file}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_file}: {e}")
            raise

    def structure_documents(self, extracted_docs: List[Document]) -> List[Document]:
        """Structure extracted text into properly formatted Documents.
        
        Args:
            extracted_docs: List of extracted Document objects
            
        Returns:
            List of structured Document objects with normalized metadata
        """
        if not extracted_docs:
            logger.warning("No documents to structure")
            return []
        
        logger.info(f"Structuring {len(extracted_docs)} documents...")
        
        structured = []
        for idx, doc in enumerate(extracted_docs):
            try:
                content = doc.page_content
                metadata = dict(doc.metadata) if doc.metadata else {}

                source = metadata.get("source", "unknown")
                page = metadata.get("page")
                page_label = metadata.get("page_label")

                metadata.update({
                    "source": source,
                    "page": str(page) if page is not None else None,
                    "page_label": page_label,
                    "university": self.university,
                })

                structured.append(Document(page_content=content, metadata=metadata))
            except Exception as e:
                logger.warning(f"Failed to structure document {idx}: {e}")
                continue
        
        logger.info(f"Successfully structured {len(structured)} documents")
        return structured
    
    def clean_markdown_content(self, content: str) -> str:
        """Clean markdown content by removing navigation, UI elements, and noise.
        
        Args:
            content: Raw markdown text from web scraping
            
        Returns:
            Cleaned markdown content focused on actual information
        """
        # Remove common noise patterns
        cleaned = content
        for pattern in self.NOISE_PATTERNS:
            cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove excessive navigation menus (keep only first occurrence)
        lines = cleaned.split('\n')
        seen_lines = {}
        filtered_lines = []
        
        for line in lines:
            # Skip empty lines in deduplication but keep them in output
            if not line.strip():
                filtered_lines.append(line)
                continue
                
            # For navigation-like lines, only keep first occurrence
            if line.strip().startswith('*') or line.strip().startswith('-'):
                line_hash = hashlib.md5(line.encode()).hexdigest()
                if line_hash in seen_lines:
                    continue
                seen_lines[line_hash] = True
            
            filtered_lines.append(line)
        
        cleaned = '\n'.join(filtered_lines)
        
        # Remove long OAuth/redirect URLs
        cleaned = re.sub(r'https?://[^\s\)]{200,}', '[URL]', cleaned)
        
        # Remove multiple consecutive blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()
    
    def classify_document_type(self, content: str, source: str, title: str) -> str:
        """Classify document type based on content and metadata.
        
        Args:
            content: Document content
            source: Source URL or file path
            title: Document title
            
        Returns:
            Document type classification
        """
        content_lower = content.lower()
        source_lower = source.lower()
        title_lower = title.lower() if title else ''
        
        if any(term in content_lower or term in source_lower or term in title_lower 
               for term in ['fee', 'fees', 'payment', 'tuition']):
            return 'fees'
        elif any(term in content_lower or term in source_lower 
                 for term in ['admission', 'apply', 'eligibility', 'entrance']):
            return 'admissions'
        elif any(term in source_lower for term in ['brochure', 'flyer']):
            return 'brochure'
        elif any(term in content_lower for term in ['contact', 'email', 'phone', 'address']):
            return 'contact'
        elif any(term in content_lower for term in ['program', 'course', 'curriculum', 'b.tech', 'm.tech']):
            return 'programs'
        elif any(term in content_lower for term in ['manual', 'guide', 'instruction']):
            return 'manual'
        else:
            return 'general'
    
    def extract_metadata_from_content(self, content: str) -> dict:
        """Extract structured metadata from content.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract dates
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b',
            r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b',
        ]
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, content, re.IGNORECASE))
        if dates:
            metadata['dates'] = list(set(dates[:5]))
        
        # Extract emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        if emails:
            metadata['emails'] = list(set(emails[:3]))
        
        # Extract phone numbers
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|\b\d{10}\b', content)
        if phones:
            metadata['phones'] = list(set(phones[:3]))
        
        return metadata
    
    def deduplicate_chunks(self, chunks: List[Document]) -> List[Document]:
        """Remove duplicate or near-duplicate chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Deduplicated list of chunks
        """
        unique_chunks = []
        buckets = {}

        def normalize_for_similarity(text: str) -> str:
            text = re.sub(r"\s+", " ", text.lower()).strip()
            text = re.sub(r"\d+", "0", text)
            return text

        def shingle_hashes(text: str, size: int = 5) -> set:
            tokens = text.split()
            if not tokens:
                return set()
            if len(tokens) <= size:
                joined = " ".join(tokens)
                return {hashlib.md5(joined.encode("utf-8")).hexdigest()}
            hashes = set()
            for i in range(len(tokens) - size + 1):
                shingle = " ".join(tokens[i:i + size])
                hashes.add(hashlib.md5(shingle.encode("utf-8")).hexdigest())
            return hashes

        def jaccard_similarity(a: set, b: set) -> float:
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        for chunk in chunks:
            normalized = normalize_for_similarity(chunk.page_content)
            content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()

            if content_hash in self._seen_content_hashes:
                logger.debug(f"Skipping duplicate chunk from {chunk.metadata.get('source', 'unknown')}")
                continue

            shingles = shingle_hashes(normalized)
            length_bucket = max(1, len(normalized) // 200)
            bucket = buckets.setdefault(length_bucket, [])

            is_near_duplicate = False
            for candidate in bucket[-200:]:
                similarity = jaccard_similarity(shingles, candidate["shingles"])
                if similarity >= 0.95:
                    is_near_duplicate = True
                    break

            if is_near_duplicate:
                logger.debug(f"Skipping near-duplicate chunk from {chunk.metadata.get('source', 'unknown')}")
                continue

            self._seen_content_hashes.add(content_hash)
            bucket.append({"shingles": shingles})
            unique_chunks.append(chunk)

        logger.info(f"Deduplicated {len(chunks)} chunks to {len(unique_chunks)} unique chunks")
        return unique_chunks

    def website_structure_documents(self, webdata_file_name: str) -> List[Document]:
        """Structure extracted text from website into properly formatted Documents.
        
        Args:
            webdata_file_name: Path to JSON file containing web-scraped data
            
        Returns:
            List of structured Document objects from website content
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON file is invalid
        """
        json_path = Path(webdata_file_name)
        
        if not json_path.exists():
            logger.error(f"Web data file not found: {webdata_file_name}")
            raise FileNotFoundError(f"Web data file not found: {webdata_file_name}")

        logger.info(f"Loading web data from {webdata_file_name}")
        
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON file {webdata_file_name}: {e}")
            raise
        
        if not raw:
            logger.warning(f"No data found in {webdata_file_name}")
            return []

        logger.info(f"Processing {len(raw)} web documents...")
        
        pages = []
        for idx, doc in enumerate(raw):
            try:
                raw_content = doc.get('metadata', {}).get('markdowntext', '')
                source = doc.get('metadata', {}).get('source', 'unknown')
                title = doc.get('metadata', {}).get('title', '')
                
                # Clean the markdown content
                cleaned_content = self.clean_markdown_content(raw_content)
                
                # Skip if content is too short after cleaning
                if len(cleaned_content.strip()) < 100:
                    logger.warning(f"Skipping document {idx} - too short after cleaning")
                    continue
                
                # Classify document type
                doc_type = self.classify_document_type(cleaned_content, source, title)
                
                # Extract additional metadata
                extracted_metadata = self.extract_metadata_from_content(cleaned_content)
                
                # Use offset to differentiate web docs from PDF docs
                doc_id = idx + 5000
                
                pages.append(
                    Document(
                        id=doc_id,
                        page_content=cleaned_content,
                        metadata={
                            "source": source,
                            "title": title,
                            "page": title,
                            "page_label": 1000,
                            "doc_type": doc_type,
                            "source_type": "web",
                            "university": self.university,
                            **extracted_metadata
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process web document {idx}: {e}")
                continue

        logger.info(f"Loaded {len(pages)} pages from website data (after cleaning)")
        
        # Split web pages into chunks with markdown-aware separators
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            # Markdown-aware separators - preserve structure
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " ", ""],
        )

        docs = splitter.split_documents(pages)
        logger.info(f"Split web data into {len(docs)} chunks (before deduplication)")
        
        # Deduplicate chunks
        docs = self.deduplicate_chunks(docs)

        structured = self.structure_documents(docs)
        return structured


def main() -> List[Document]:
    """Main function to process all documents and create knowledge base.
    
    Complete pipeline with checkpointing:
    1. Web scraping
    2. Attachment downloading
    3. PDF processing
    4. Web data processing
    5. Combining and final save
    
    Returns:
        List of all structured documents (web + PDF)
    """
    logger.info("="*80)
    logger.info("STARTING COMPLETE KNOWLEDGE BASE CREATION PIPELINE")
    logger.info("="*80)
    
    obj = KnowledgeBaseCreation()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Clear previous table CSVs to avoid duplicates
    obj.clear_tables_checkpoint_dir()
    
    # ========== STEP 1: WEB SCRAPING ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 1: WEB SCRAPING")
    logger.info("="*80)
    
    urls_to_scrape = [
        "https://www.coeptech.ac.in/admissions/undergraduate/",
        "https://www.coeptech.ac.in/about-us/about-university/",
        "https://www.coeptech.ac.in/hostel/hostel-admissions/",
        "https://www.coeptech.ac.in/hostel/rules-and-regulations/",
        "https://www.coeptech.ac.in/student-corner/student-services/student-helpline/",
        "https://www.coeptech.ac.in/student-corner/student-clubs/",
        "https://www.coeptech.ac.in/facilities/facilities-manager/facilities-for-differently-abled-individuals/",
        "https://www.coeptech.ac.in/useful-links/university-sections/",
        "https://www.coeptech.ac.in/admissions/undergraduate/first-year-admissions/",
        "https://www.coeptech.ac.in/admissions/undergraduate/direct-second-year-admission/",
        "https://www.coeptech.ac.in/admissions/undergraduate/working-professional/",
        "https://mtech2025.coeptech.ac.in/StaticPages/HomePage",
        "https://www.coeptech.ac.in/admissions/post-graduate/",
        "https://www.coeptech.ac.in/admissions/ph-d/",
        "https://www.coeptech.ac.in/admissions/mba/"
    ]
    
    try:
        logger.info(f"Scraping {len(urls_to_scrape)} URLs...")
        data:list = []
        for idx, url in enumerate(urls_to_scrape):
            logger.info(f"  [{idx}/{len(urls_to_scrape)}] {url}")
            temp_data = WebBasedLoader.load_html_markdown_from_url(url)
            if temp_data:
                data.extend(temp_data)
            if idx < len(urls_to_scrape):
                sleep(2)  # Be respectful to the server



        # Convert documents to serializable shape and write to file
        serializable = WebBasedLoader.documents_to_serializable(data)
        out_path = "extracted_text_data/admissions_data.json"
        WebBasedLoader.write_json_atomic(out_path, serializable)
        logger.info("Wrote %d documents to %s", len(serializable), out_path)
        

        # Save web scraping checkpoint
        web_data_path = Path("extracted_text_data/admissions_data.json")
        if web_data_path.exists():
            with open(web_data_path, 'r', encoding='utf-8') as f:
                web_data = json.load(f)
            obj.save_checkpoint(web_data, "01_web_scraping", timestamp)
            logger.info(f"✅ Web scraping complete: {len(web_data)} pages extracted")
        else:
            logger.warning("Web scraping completed but admissions_data.json not found")
            web_data = []
    except Exception as e:
        logger.error(f"Failed to scrape websites: {e}")
        web_data = []
    
    # ========== STEP 2: DOWNLOAD ATTACHMENTS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DOWNLOADING ATTACHMENTS")
    logger.info("="*80)
    
    try:
        json_path = Path("extracted_text_data/admissions_data.json")
        if json_path.exists():
            logger.info("Scanning web data for downloadable attachments...")
            downloader = AttachmentDownloader()
            downloader.download_all_attachments(json_path)
            
            # Reload and checkpoint the updated JSON with attachment metadata
            with open(json_path, 'r', encoding='utf-8') as f:
                web_data_with_attachments = json.load(f)
            obj.save_checkpoint(web_data_with_attachments, "02_attachments_downloaded", timestamp)
            logger.info("✅ Attachment downloading complete")
        else:
            logger.warning("Cannot download attachments - admissions_data.json not found")
    except Exception as e:
        logger.error(f"Failed to download attachments: {e}")
    
    # ========== STEP 3: PROCESS PDF DOCUMENTS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PROCESSING PDF DOCUMENTS")
    logger.info("="*80)
    
    try:
        logger.info("Extracting text from PDFs in attachments/ directory...")
        extracted_docs = obj.extract_multiple_pdfs("attachments/")
        
        # Save raw extracted PDF documents
        obj.save_checkpoint(extracted_docs, "03_pdf_extraction_raw", timestamp)

        merit_table_docs = []
        filtered_pdf_docs = []
        for doc in extracted_docs:
            doc_type = (doc.metadata or {}).get("doc_type")
            filename = (doc.metadata or {}).get("filename", "")
            if doc_type == "merit_list" or "merit" in filename.lower():
                merit_table_docs.append(doc)
            else:
                filtered_pdf_docs.append(doc)

        if merit_table_docs:
            obj.save_checkpoint(merit_table_docs, "04_pdf_tables", timestamp)
            logger.info(f"✅ Saved {len(merit_table_docs)} merit list tables to separate checkpoint")

        # Structure PDF documents (including non-merit tables)
        structured_pdf_docs = obj.structure_documents(filtered_pdf_docs)
        obj.save_checkpoint(structured_pdf_docs, "04_pdf_structured", timestamp)

        logger.info(f"✅ PDF processing complete: {len(structured_pdf_docs)} documents")
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}")
        structured_pdf_docs = []
    
    # ========== STEP 4: PROCESS WEBSITE DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PROCESSING WEBSITE DATA")
    logger.info("="*80)
    
    try:
        logger.info("Cleaning and structuring web data...")
        web_structured_docs = obj.website_structure_documents(
            "extracted_text_data/admissions_data.json"
        )
        obj.save_checkpoint(web_structured_docs, "05_web_structured", timestamp)
        logger.info(f"✅ Web data processing complete: {len(web_structured_docs)} documents")
    except Exception as e:
        logger.error(f"Failed to process web data: {e}")
        web_structured_docs = []
    
    # ========== STEP 5: COMBINE ALL DOCUMENTS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 5: COMBINING ALL DOCUMENTS")
    logger.info("="*80)
    
    all_docs = web_structured_docs + structured_pdf_docs
    logger.info(f"Total documents: {len(all_docs)}")
    logger.info(f"  - Web documents: {len(web_structured_docs)}")
    logger.info(f"  - PDF documents: {len(structured_pdf_docs)}")
    
    # Final cross-source deduplication
    all_docs = obj.deduplicate_chunks(all_docs)
    logger.info(f"Total documents after final deduplication: {len(all_docs)}")

    # Save final combined documents
    obj.save_checkpoint(all_docs, "06_final_combined", timestamp)
    
    # Also save a human-readable summary
    summary = {
        "timestamp": timestamp,
        "total_documents": len(all_docs),
        "web_documents": len(web_structured_docs),
        "pdf_documents": len(structured_pdf_docs),
        "urls_scraped": len(urls_to_scrape),
        "processing_steps": [
            "01_web_scraping",
            "02_attachments_downloaded",
            "03_pdf_extraction_raw",
            "04_pdf_structured",
            "05_web_structured",
            "06_final_combined"
        ]
    }
    
    summary_file = obj.checkpoint_dir / f"summary_{timestamp}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info("="*80)
    logger.info(f"All checkpoints saved in: {obj.checkpoint_dir}")
    logger.info(f"Summary file: {summary_file}")
    
    return all_docs


def create_embeddings_in_batches(
    documents: List[Document],
    batch_size: int = 70,
    sleep_time: int = 60
) -> None:
    """Create embeddings for documents in batches to avoid rate limits.
    
    Args:
        documents: List of documents to embed
        batch_size: Number of documents per batch
        sleep_time: Sleep duration between batches (seconds)
    """
    logger.info(f"Creating embeddings for {len(documents)} documents in batches of {batch_size}")
    
    embed_gen = EmbeddingsGeneration()
    
    # Process first batch
    first_batch_size = min(batch_size, len(documents))
    logger.info(f"Processing first batch of {first_batch_size} documents...")
    
    try:
        vector_store = embed_gen.openai_embeddings_generation(
            docs=documents[:first_batch_size]
        )
        logger.info(f"✅ First batch complete")
    except Exception as e:
        logger.error(f"Failed to process first batch: {e}")
        raise
    
    # Process remaining batches
    if len(documents) > batch_size:
        remaining = len(documents) - batch_size
        num_batches = (remaining + batch_size - 1) // batch_size
        
        logger.info(f"Processing {num_batches} additional batches...")
        
        for i in range(num_batches):
            start_idx = batch_size + (i * batch_size)
            end_idx = min(start_idx + batch_size, len(documents))
            
            logger.info(f"Batch {i+2}/{num_batches+1}: Documents {start_idx} to {end_idx}")
            logger.info(f"Sleeping for {sleep_time} seconds to avoid rate limits...")
            sleep(sleep_time)
            
            try:
                batch_docs = documents[start_idx:end_idx]
                embed_gen.add_documents(batch_docs)
                logger.info(f"✅ Batch {i+2} complete")
            except Exception as e:
                logger.error(f"Failed to process batch {i+2}: {e}")
                continue
    
    logger.info("✅ All embeddings created successfully!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create knowledge base
    structured_docs = main()

    print("\n" + "="*80)
    print(f"📚 Knowledge Base Summary")
    print("="*80)
    print(f"Total documents: {len(structured_docs)}")
    
    if structured_docs:
        print(f"\nFirst document preview:")
        print(f"Content: {structured_docs[0].page_content[:200]}...")
        print(f"Metadata: {structured_docs[0].metadata}")
    
    print("\n" + "="*80)
    print("💡 Next Steps:")
    print("="*80)
    print("To create embeddings, uncomment the code below and run:")
    print("  python -m institute_qna.data_preprocess.knoweldge_base_creation")
    print()
    
    # Uncomment to create embeddings
    # print("\n🔄 Creating embeddings...")
    create_embeddings_in_batches(structured_docs, batch_size=70, sleep_time=60)
