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
from institute_qna.data_preprocess.knoweldge_base_creation import main, create_embeddings_in_batches
from pathlib import Path
import json

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
    """Display information about saved checkpoints."""
    checkpoint_dir = Path("extracted_text_data/checkpoints")
    
    if not checkpoint_dir.exists():
        logger.warning("No checkpoint directory found")
        return
    
    checkpoints = sorted(checkpoint_dir.glob("*.json"))
    
    if not checkpoints:
        logger.info("No checkpoints found")
        return
    
    print("\n" + "="*80)
    print("SAVED CHECKPOINTS")
    print("="*80)
    
    for cp in checkpoints:
        size = cp.stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"\n📄 {cp.name}")
        print(f"   Size: {size_mb:.2f} MB")
        
        # Try to show document count if it's a list
        try:
            with open(cp, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"   Items: {len(data)}")
            elif isinstance(data, dict) and 'total_documents' in data:
                print(f"   Total documents: {data['total_documents']}")
        except:
            pass
    
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
        print("  1. Run: create_embeddings_in_batches(structured_docs)")
        print("  2. Or uncomment the line below in this script")
        print()
        
        # Uncomment to create embeddings immediately
        # user_input = input("Create embeddings now? (y/n): ")
        # if user_input.lower() == 'y':
        #     logger.info("Creating embeddings...")
        #     create_embeddings_in_batches(structured_docs, batch_size=70, sleep_time=60)
        
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
