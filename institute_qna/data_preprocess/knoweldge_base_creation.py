"""Knowledge base creation pipeline orchestration.

Main flow lives in this file. Reusable class and helper logic are located in
`knoweldge_base_creation_utils.py`.
"""

import logging
from datetime import datetime
from time import sleep
from typing import List

from langchain_core.documents import Document

from institute_qna.data_extraction.download_attachment import AttachmentDownloader
from institute_qna.data_extraction.webscrapper import WebBasedLoader
from institute_qna.data_preprocess.knoweldge_base_creation_utils import KnowledgeBaseCreation

logger = logging.getLogger(__name__)


def main(extraction_method: str = "azure") -> List[Document]:
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
    # urls_to_scrape = [
    #     "https://mitaoe.ac.in/admission.php",
    #     "https://mitaoe.ac.in/first-year-BTech.php",
    #     "https://mitaoe.ac.in/MTech.php",
    #     "https://mitaoe.ac.in/contact-us.php"

    # ]
    
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
    except Exception as e:
        logger.error(f"Failed to scrape websites: {e}")
        web_data = []
    
    # ========== STEP 2: DOWNLOAD ATTACHMENTS ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 2: DOWNLOADING ATTACHMENTS")
    logger.info("="*80)
    
    try:
        if web_data:
            logger.info("Scanning web data for downloadable attachments (in-memory)...")
            web_data_with_attachments = AttachmentDownloader.augment_documents_with_attachment_links(web_data)
            obj.save_checkpoint(web_data_with_attachments, "02_attachments_downloaded", timestamp)
            logger.info("✅ Attachment link enrichment complete")
        else:
            logger.warning("Cannot scan attachments - no web data available")
            web_data_with_attachments = []
    except Exception as e:
        logger.error(f"Failed to download attachments: {e}")
        web_data_with_attachments = web_data
    
    # ========== STEP 3: PROCESS PDF DOCUMENTS ==========
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
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}")
        structured_pdf_docs = []
    
    # ========== STEP 4: PROCESS WEBSITE DATA ==========
    logger.info("\n" + "="*80)
    logger.info("STEP 4: PROCESSING WEBSITE DATA")
    logger.info("="*80)
    
    try:
        logger.info("Cleaning and structuring web data...")
        web_structured_docs = obj.website_structure_documents(web_data_with_attachments)
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
        "run_blob_folder": obj.run_blob_folder,
        "checkpoints_blob_folder": obj.checkpoints_blob_folder,
        "extraction_processing_blob_folder": f"{obj.run_blob_folder}/extraction_processing",
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
    logger.info(f"Summary blob path: {summary_blob_path}")
    
    return all_docs


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
    print("To create embeddings, run dedicated vector-store modules:")
    print("  python -m institute_qna.data_preprocess.knoweldge_base_creation")
    print("  - Chroma: use institute_qna.data_preprocess.chroma_vector_store")
    print("  - Azure AI Search: use institute_qna.data_preprocess.azure_ai_search_store")
