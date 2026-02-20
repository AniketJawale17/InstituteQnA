"""Knowledge base creation pipeline orchestration.

Main flow lives in this file. Reusable class and helper logic are located in
`knoweldge_base_creation_utils.py`.
"""

import logging
import os
from datetime import datetime
from time import sleep
from typing import Callable, List, Optional

from langchain_core.documents import Document

from institute_qna.data_extraction.download_attachment import AttachmentDownloader
from institute_qna.data_extraction.webscrapper import WebBasedLoader
from institute_qna.logging_config import configure_logging
from institute_qna.data_preprocess.knoweldge_base_creation_utils import KnowledgeBaseCreation
from institute_qna.data_preprocess.vector_store_ingestion import (
    store_documents_in_vector_backend_from_blob,
)

logger = logging.getLogger(__name__)


def _emit_progress(progress_callback: Optional[Callable[[str], None]], message: str) -> None:
    """Emit human-friendly progress updates to caller if provided."""
    if progress_callback:
        progress_callback(message)


def main(
    extraction_method: str = "azure",
    progress_callback: Optional[Callable[[str], None]] = None,
) -> List[Document]:
    """Main function to process all documents and create knowledge base.
    
    Complete pipeline with checkpointing:
    1. Web scraping
    2. Attachment downloading
    3. PDF processing
    4. Web data processing
    5. Combining and final save
    
    Args:
        extraction_method: PDF extraction method - 'opensource' (default) or 'azure'
    
    Returns:
        List of all structured documents (web + PDF)
    """
    logger.info("="*80)
    logger.info("STARTING COMPLETE KNOWLEDGE BASE CREATION PIPELINE")
    logger.info(f"PDF Extraction Method: {extraction_method.upper()}")
    logger.info("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obj = KnowledgeBaseCreation(
        university="pune",
        extraction_method=extraction_method,
        run_timestamp=timestamp,
    )

    # ========== STEP 1: WEB SCRAPING ==========
    _emit_progress(progress_callback, "Step 1/6: Web scraping started")
    logger.info("\n" + "="*80)
    logger.info("STEP 1: WEB SCRAPING")
    logger.info("="*80)
    
    urls_to_scrape = [
        "https://www.coeptech.ac.in/admissions/undergraduate/"
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
    
    urls_to_scrape_2 = [
        "https://mitaoe.ac.in/admission.php",
        "https://mitaoe.ac.in/first-year-BTech.php",
        "https://mitaoe.ac.in/MTech.php",
        "https://mitaoe.ac.in/contact-us.php"

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



        # Convert documents to serializable shape in-memory
        serializable = WebBasedLoader.documents_to_serializable(data)
        web_data = serializable
        obj.save_checkpoint(web_data, "01_web_scraping", timestamp)
        logger.info(f"✅ Web scraping complete: {len(web_data)} pages extracted")
        _emit_progress(progress_callback, f"Step 1/6 complete: Web pages extracted ({len(web_data)})")
    except Exception as e:
        logger.error(f"Failed to scrape websites: {e}")
        web_data = []
        _emit_progress(progress_callback, "Step 1/6 failed: Web scraping")
    
    # ========== STEP 2: DOWNLOAD ATTACHMENTS ==========
    _emit_progress(progress_callback, "Step 2/6: Attachment download started")
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DOWNLOADING ATTACHMENTS")
    logger.info("="*80)
    
    try:
        if web_data:
            logger.info("Scanning web data for attachments and uploading files to Azure Blob...")
            web_data_with_attachments = AttachmentDownloader().download_attachments_from_documents(web_data)
            obj.save_checkpoint(web_data_with_attachments, "02_attachments_downloaded", timestamp)
            logger.info("✅ Attachment download/upload step complete")
            _emit_progress(progress_callback, "Step 2/6 complete: Attachments processed")
        else:
            logger.warning("Cannot scan attachments - no web data available")
            web_data_with_attachments = []
            _emit_progress(progress_callback, "Step 2/6 skipped: No web data available")
    except Exception as e:
        logger.error(f"Failed to download attachments: {e}")
        web_data_with_attachments = web_data
        _emit_progress(progress_callback, "Step 2/6 failed: Attachment processing")
    
    # ========== STEP 3: PROCESS PDF DOCUMENTS ==========
    _emit_progress(progress_callback, "Step 3/6: PDF extraction started")
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PROCESSING PDF DOCUMENTS")
    logger.info("="*80)
    
    try:
        logger.info("Extracting text from PDF attachment URLs without local file writes...")
        attachment_urls = []
        for item in web_data_with_attachments:
            metadata = item.get("metadata", {}) if isinstance(item, dict) else {}
            links = metadata.get("attachments", [])
            for link in links:
                link_lower = str(link).lower()
                if ".pdf" in link_lower:
                    attachment_urls.append(link)

        extracted_docs = obj.extract_multiple_pdfs_from_urls(attachment_urls)
        
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
        _emit_progress(progress_callback, f"Step 3/6 complete: PDF documents processed ({len(structured_pdf_docs)})")
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}")
        structured_pdf_docs = []
        _emit_progress(progress_callback, "Step 3/6 failed: PDF processing")
    
    # ========== STEP 4: PROCESS WEBSITE DATA ==========
    _emit_progress(progress_callback, "Step 4/6: Website data structuring started")
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PROCESSING WEBSITE DATA")
    logger.info("="*80)
    
    try:
        logger.info("Cleaning and structuring web data...")
        web_structured_docs = obj.website_structure_documents(web_data_with_attachments)
        obj.save_checkpoint(web_structured_docs, "05_web_structured", timestamp)
        logger.info(f"✅ Web data processing complete: {len(web_structured_docs)} documents")
        _emit_progress(progress_callback, f"Step 4/6 complete: Website documents structured ({len(web_structured_docs)})")
    except Exception as e:
        logger.error(f"Failed to process web data: {e}")
        web_structured_docs = []
        _emit_progress(progress_callback, "Step 4/6 failed: Website data structuring")
    
    # ========== STEP 5: COMBINE ALL DOCUMENTS ==========
    _emit_progress(progress_callback, "Step 5/6: Combining and deduplicating documents")
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
    final_processed_blob_path = obj.save_final_processed_documents(all_docs)
    
    # Also save a human-readable summary
    summary = {
        "timestamp": timestamp,
        "run_blob_folder": obj.run_blob_folder,
        "checkpoints_blob_folder": obj.checkpoints_blob_folder,
        "extraction_processing_blob_folder": f"{obj.run_blob_folder}/extraction_processing",
        "final_processed_blob_path": final_processed_blob_path,
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
    
    summary_blob_path = obj.save_checkpoint(summary, "summary", timestamp)
    _emit_progress(progress_callback, f"Step 5/6 complete: Final combined documents ({len(all_docs)})")

    # ========== STEP 6: VECTOR STORE INGESTION ==========
    _emit_progress(progress_callback, "Step 6/6: Vector store ingestion started")
    logger.info("\n" + "="*80)
    logger.info("STEP 6: VECTOR STORE INGESTION")
    logger.info("="*80)
    vector_backend = os.getenv("VECTOR_BACKEND", "azure_ai_search")
    try:
        store_documents_in_vector_backend_from_blob(
            blob_path=final_processed_blob_path,
            backend=vector_backend,
        )
        logger.info(
            "✅ Vector store ingestion complete from blob: %s (backend=%s)",
            final_processed_blob_path,
            vector_backend,
        )
        _emit_progress(progress_callback, f"Step 6/6 complete: Vector ingestion finished ({vector_backend})")
    except Exception as e:
        logger.error("Failed vector store ingestion from blob: %s", e)
        _emit_progress(progress_callback, "Step 6/6 failed: Vector store ingestion")
        raise
    
    logger.info("="*80)
    logger.info("PIPELINE COMPLETE ✅")
    logger.info("="*80)
    logger.info(
        "Run data uploaded to Azure Blob container '%s' under folder '%s/'",
        obj.azure_blob_container_name,
        obj.run_blob_folder,
    )
    logger.info("Checkpoints folder: %s", obj.checkpoints_blob_folder)
    logger.info("Extraction processing folder: %s/extraction_processing", obj.run_blob_folder)
    logger.info("Final processed data path: %s", final_processed_blob_path)
    logger.info("Vector ingestion backend: %s", vector_backend)
    logger.info(f"Summary blob path: {summary_blob_path}")
    
    return all_docs


if __name__ == "__main__":
    configure_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=None,
        console=False,
    )

    def _console_progress(message: str) -> None:
        print(f"[PROGRESS] {datetime.now().strftime('%H:%M:%S')} - {message}", flush=True)

    _console_progress("Knowledge base creation started")
    
    # Create knowledge base
    structured_docs = main(progress_callback=_console_progress)
    _console_progress("Knowledge base creation completed")
    logger.info("%s", "=" * 80)
    logger.info("📚 Knowledge Base Summary")
    logger.info("%s", "=" * 80)
    logger.info("Total documents: %d", len(structured_docs))

    if structured_docs:
        logger.info("First document preview")
        logger.info("Content: %s...", structured_docs[0].page_content[:200])
        logger.info("Metadata: %s", structured_docs[0].metadata)

    logger.info("%s", "=" * 80)
