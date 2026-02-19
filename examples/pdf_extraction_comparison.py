"""
PDF Extraction Methods Comparison Example

This script demonstrates how to use both open-source and Azure Document Intelligence
methods for PDF extraction, comparing their output and performance.
"""

import time
import os
from pathlib import Path
from institute_qna.data_preprocess import PDFTextExtractor


def compare_extraction_methods(pdf_path: str):
    """Compare open-source and Azure extraction methods.
    
    Args:
        pdf_path: Path to PDF file to analyze
    """
    print("="*80)
    print(f"PDF Extraction Methods Comparison")
    print(f"File: {pdf_path}")
    print("="*80)
    
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return
    
    # Method 1: Open Source (PyPDF + pdfplumber)
    print("\n" + "="*80)
    print("Method 1: Open Source (PyPDF + pdfplumber)")
    print("="*80)
    
    start_time = time.time()
    try:
        extractor_os = PDFTextExtractor(
            pdf_path=pdf_path,
            extraction_method="opensource",
            chunk_size=2000,
            chunk_overlap=200
        )
        docs_os = extractor_os.extract_text_from_text_pdf()
        os_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"📊 Results:")
        print(f"   - Total chunks: {len(docs_os)}")
        print(f"   - Processing time: {os_time:.2f}s")
        print(f"   - Cost: $0 (Free)")
        
        # Count documents with tables
        docs_with_tables = [d for d in docs_os if d.metadata.get('has_table', False)]
        print(f"   - Documents with tables: {len(docs_with_tables)}")
        
        if docs_os:
            print(f"\n📝 Sample Output (first 300 chars):")
            print("-" * 80)
            print(docs_os[0].page_content[:300])
            print("\n📌 Metadata:")
            for key, value in docs_os[0].metadata.items():
                if key != 'page_content':
                    print(f"   - {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Method 2: Azure Document Intelligence
    print("\n" + "="*80)
    print("Method 2: Azure Document Intelligence")
    print("="*80)
    
    # Check if Azure credentials are available
    if not os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT") or not os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY"):
        print("⚠️  Azure credentials not found!")
        print("   Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY")
        print("   Skipping Azure comparison...")
        print("\n💡 To use Azure method:")
        print("   1. Create Azure Document Intelligence resource")
        print("   2. Add credentials to .env file")
        return
    
    start_time = time.time()
    try:
        extractor_azure = PDFTextExtractor(
            pdf_path=pdf_path,
            extraction_method="azure",
            chunk_size=2000,
            chunk_overlap=200
        )
        docs_azure = extractor_azure.extract_text_from_text_pdf()
        azure_time = time.time() - start_time
        
        print(f"✅ Success!")
        print(f"📊 Results:")
        print(f"   - Total chunks: {len(docs_azure)}")
        print(f"   - Processing time: {azure_time:.2f}s")
        
        # Estimate cost (approximate)
        num_pages = len(docs_azure) // 2  # Rough estimate
        estimated_cost = num_pages * 0.001  # $0.001 per page (example rate)
        print(f"   - Estimated cost: ~${estimated_cost:.3f}")
        
        # Count documents with tables
        docs_with_tables = [d for d in docs_azure if d.metadata.get('has_table', False)]
        print(f"   - Documents with tables: {len(docs_with_tables)}")
        
        if docs_azure:
            print(f"\n📝 Sample Output (first 300 chars):")
            print("-" * 80)
            print(docs_azure[0].page_content[:300])
            print("\n📌 Metadata:")
            for key, value in docs_azure[0].metadata.items():
                if key != 'page_content':
                    print(f"   - {key}: {value}")
        
        # Comparison
        print("\n" + "="*80)
        print("📈 Comparison Summary")
        print("="*80)
        print(f"Open Source chunks: {len(docs_os)}")
        print(f"Azure chunks: {len(docs_azure)}")
        print(f"Speed difference: {abs(azure_time - os_time):.2f}s")
        print(f"Faster method: {'Open Source' if os_time < azure_time else 'Azure'}")
        
    except ImportError:
        print("❌ Azure Document Intelligence SDK not installed")
        print("   Install with: pip install azure-ai-documentintelligence")
    except Exception as e:
        print(f"❌ Error: {e}")


def extract_multiple_pdfs_comparison(folder_path: str, method: str = "opensource"):
    """Extract multiple PDFs using specified method.
    
    Args:
        folder_path: Path to folder containing PDFs
        method: Extraction method ('opensource' or 'azure')
    """
    print("="*80)
    print(f"Batch PDF Processing - {method.upper()} Method")
    print(f"Folder: {folder_path}")
    print("="*80)
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"❌ Error: Folder not found: {folder_path}")
        return
    
    pdf_files = list(folder.glob("*.pdf"))
    if not pdf_files:
        print(f"❌ Error: No PDF files found in {folder_path}")
        return
    
    print(f"\n📁 Found {len(pdf_files)} PDF files")
    
    start_time = time.time()
    try:
        extractor = PDFTextExtractor(
            extraction_method=method
        )
        all_docs = extractor.extract_multiple_pdfs(folder_path)
        processing_time = time.time() - start_time
        
        print(f"\n✅ Processing complete!")
        print(f"📊 Results:")
        print(f"   - Total document chunks: {len(all_docs)}")
        print(f"   - Processing time: {processing_time:.2f}s")
        print(f"   - Average per PDF: {processing_time / len(pdf_files):.2f}s")
        
        # Count by document type
        doc_types = {}
        for doc in all_docs:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"\n📑 Document Types:")
        for doc_type, count in doc_types.items():
            print(f"   - {doc_type}: {count} chunks")
        
        # Count tables
        docs_with_tables = [d for d in all_docs if d.metadata.get('has_table', False)]
        print(f"\n📊 Tables detected: {len(docs_with_tables)} chunks")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    import sys
    
    print("\n🔍 PDF Extraction Methods Demo\n")
    
    # Check if PDF path provided
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        compare_extraction_methods(pdf_path)
    else:
        # Default examples
        print("Usage: python pdf_extraction_comparison.py <pdf_path>")
        print("\nExample commands:")
        print("  python examples/pdf_extraction_comparison.py attachments/brochure.pdf")
        print("  python examples/pdf_extraction_comparison.py attachments/fees.pdf")
        print("\n" + "="*80)
        print("Demo: Batch Processing")
        print("="*80)
        
        # Try batch processing if attachments folder exists
        if Path("attachments").exists():
            extract_multiple_pdfs_comparison("attachments", method="opensource")
        else:
            print("⚠️  'attachments' folder not found. Skipping demo.")
            print("   Create an 'attachments' folder and add PDF files to test.")
