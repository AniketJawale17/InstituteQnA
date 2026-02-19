#!/usr/bin/env python3
"""Complete Knowledge Base Creation Pipeline with Checkpointing.

This script runs the entire pipeline:
1. Web scraping from university website
2. Downloading PDF attachments
3. Processing PDF documents
4. Cleaning and structuring web data
5. Combining all documents
6. Creating embeddings

Each step saves checkpoints to prevent data loss.
"""

import logging
from institute_qna.data_preprocess.knoweldge_base_creation import main
from institute_qna.data_preprocess.chroma_vector_store import create_embeddings_in_batches
from institute_qna.data_preprocess.vector_store_ingestion import store_documents_in_vector_backend
from pathlib import Path
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline_run.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def display_checkpoint_info():
    """Display Azure Blob checkpoint destination details."""
    container = os.getenv("AZURE_STORAGE_CONTAINER_NAME", "qna-checkpoints")
    runs_prefix = os.getenv("AZURE_RUNS_BLOB_PREFIX", "processing_runs")
    print("\n" + "="*80)
    print("CHECKPOINT DESTINATION")
    print("="*80)
    print("📦 Azure Blob Storage")
    print(f"   Container: {container}")
    print(f"   Run folder format: {runs_prefix}/<timestamp>/")
    print("   Checkpoints folder: <run>/checkpoints/")
    print("   Example: processing_runs/20260219_220501/checkpoints/06_final_combined_20260219_220501.json")
    print("\n" + "="*80)


def main_pipeline():
    """Run the complete pipeline."""
    try:
        # Step 1-5: Complete document processing with checkpointing
        logger.info("Starting complete pipeline...")
        structured_docs = main()
        
        # Display summary
        print("\n" + "="*80)
        print("📚 KNOWLEDGE BASE SUMMARY")
        print("="*80)
        print(f"Total documents: {len(structured_docs)}")
        
        if structured_docs:
            print(f"\n📄 First document preview:")
            print(f"Content: {structured_docs[0].page_content[:200]}...")
            print(f"Metadata: {structured_docs[0].metadata}")
        
        # Show checkpoint information
        display_checkpoint_info()
        
        # Step 6: Create embeddings (optional)
        print("\n" + "="*80)
        print("💡 NEXT STEPS")
        print("="*80)
        print("To create embeddings, you can:")
        print("  1. Azure AI Search (default): store_documents_in_vector_backend(structured_docs)")
        print("  2. Chroma (optional): create_embeddings_in_batches(structured_docs)")
        print("  3. Or uncomment one option below in this script")
        print()
        
        # Uncomment to create embeddings immediately
        # user_input = input("Create embeddings now? (y/n): ")
        # if user_input.lower() == 'y':
        #     logger.info("Creating embeddings...")
        #     create_embeddings_in_batches(structured_docs, batch_size=70, sleep_time=60)
        #     store_documents_in_vector_backend(structured_docs, backend="azure_ai_search")
        
        return structured_docs
        
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        
        # Even if pipeline fails, show what checkpoints were saved
        print("\n⚠️ Pipeline encountered an error, but checkpoints may have been saved:")
        display_checkpoint_info()
        
        raise


if __name__ == "__main__":
    # Ensure logs directory exists
    Path("logs").mkdir(exist_ok=True)
    
    # Run pipeline
    structured_docs = main_pipeline()
    
    print("\n✅ Pipeline execution complete!")
    print(f"Check logs/pipeline_run.log for detailed logs")
