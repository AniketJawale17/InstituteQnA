"""Knowledge Base Creation Module.

This module structures extracted text from PDFs and websites into a format 
suitable for knowledge base creation and vector embedding.
"""

from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from institute_qna.data_preprocess.embedding_generation import EmbeddingsGeneration
from langchain_core.documents import Document 
from pathlib import Path
from typing import List
import logging
from time import sleep
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class KnowledgeBaseCreation(PDFTextExtractor):
    """Generates Knowledge base through structuring extracted text.
    
    This class processes both PDF documents and web-scraped data, structuring
    them into a unified format suitable for embedding and retrieval.
    """
    
    def __init__(self):
        """Initialize the Knowledge Base Creation class."""
        super().__init__()
        self.structured_documents: List[Document] = []

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
                source = doc.metadata.get('source', 'unknown')
                page = doc.metadata.get('page')
                page_label = doc.metadata.get('page_label')
                
                structured.append(
                    Document(
                        id=idx,
                        page_content=content,
                        metadata={
                            "source": source,
                            "page": str(page) if page is not None else None,
                            "page_label": page_label
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to structure document {idx}: {e}")
                continue
        
        logger.info(f"Successfully structured {len(structured)} documents")
        return structured

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
                content = doc.get('metadata', {}).get('markdowntext', '')
                source = doc.get('metadata', {}).get('source', 'unknown')
                title = doc.get('metadata', {}).get('title')
                
                # Use offset to differentiate web docs from PDF docs
                doc_id = idx + 5000
                
                pages.append(
                    Document(
                        id=doc_id,
                        page_content=content,
                        metadata={
                            "source": source,
                            "page": title,
                            "page_label": 1000
                        }
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to process web document {idx}: {e}")
                continue

        logger.info(f"Loaded {len(pages)} pages from website data")
        
        # Split web pages into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        docs = splitter.split_documents(pages)
        logger.info(f"Split web data into {len(docs)} chunks")

        structured = self.structure_documents(docs)
        return structured


def main() -> List[Document]:
    """Main function to process all documents and create knowledge base.
    
    Returns:
        List of all structured documents (web + PDF)
    """
    logger.info("Starting knowledge base creation...")
    
    obj = KnowledgeBaseCreation()
    
    # Process PDF documents
    try:
        logger.info("Processing PDF documents...")
        extracted_docs = obj.extract_multiple_pdfs("attachments/")
        structured_docs = obj.structure_documents(extracted_docs)
        logger.info(f"Processed {len(structured_docs)} PDF documents")
    except Exception as e:
        logger.error(f"Failed to process PDFs: {e}")
        structured_docs = []
    
    # Process website data
    try:
        logger.info("Processing website data...")
        web_structured_docs = obj.website_structure_documents(
            "extracted_text_data/admissions_data.json"
        )
        logger.info(f"Processed {len(web_structured_docs)} web documents")
    except Exception as e:
        logger.error(f"Failed to process web data: {e}")
        web_structured_docs = []
    
    # Combine all documents
    all_docs = web_structured_docs + structured_docs
    logger.info(f"Total documents in knowledge base: {len(all_docs)}")
    
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
    # create_embeddings_in_batches(structured_docs, batch_size=70, sleep_time=60)
