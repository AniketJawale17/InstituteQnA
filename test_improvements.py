"""Test script to verify document processing improvements."""

import logging
from institute_qna.data_preprocess.knoweldge_base_creation import KnowledgeBaseCreation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_improvements():
    """Test the improved document processing pipeline."""
    print("\n" + "="*80)
    print("🧪 Testing Document Processing Improvements")
    print("="*80)
    
    kb = KnowledgeBaseCreation()
    
    # Test 1: Web data cleaning and classification
    print("\n📄 Test 1: Processing Web Data with Cleaning & Classification")
    print("-" * 80)
    try:
        web_docs = kb.website_structure_documents("extracted_text_data/admissions_data.json")
        print(f"✅ Processed {len(web_docs)} web documents")
        
        # Show document types
        doc_types = {}
        for doc in web_docs:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
        
        print(f"\n📊 Document Types Distribution:")
        for doc_type, count in sorted(doc_types.items()):
            print(f"   - {doc_type}: {count} chunks")
        
        # Show sample of cleaned content
        if web_docs:
            print(f"\n📝 Sample Cleaned Content (first 500 chars):")
            print("-" * 80)
            print(web_docs[0].page_content[:500])
            print("\n📌 Metadata:")
            for key, value in web_docs[0].metadata.items():
                if key != 'page_content':
                    print(f"   - {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error processing web data: {e}")
    
    # Test 2: PDF extraction with tables
    print("\n\n📄 Test 2: Processing PDFs with Table Extraction")
    print("-" * 80)
    try:
        pdf_docs = kb.extract_multiple_pdfs("attachments/")
        print(f"✅ Processed PDF documents: {len(pdf_docs)} chunks")
        
        # Show PDFs with tables
        docs_with_tables = [d for d in pdf_docs if d.metadata.get('has_table', False)]
        print(f"📊 Chunks with tables: {len(docs_with_tables)}")
        
        # Show document types
        pdf_types = {}
        for doc in pdf_docs:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            pdf_types[doc_type] = pdf_types.get(doc_type, 0) + 1
        
        print(f"\n📊 PDF Document Types:")
        for doc_type, count in sorted(pdf_types.items()):
            print(f"   - {doc_type}: {count} chunks")
        
        # Show sample with table
        if docs_with_tables:
            print(f"\n📝 Sample PDF Chunk with Table (first 500 chars):")
            print("-" * 80)
            print(docs_with_tables[0].page_content[:500])
    
    except Exception as e:
        print(f"❌ Error processing PDFs: {e}")
    
    # Test 3: Combined knowledge base
    print("\n\n📚 Test 3: Combined Knowledge Base Statistics")
    print("-" * 80)
    try:
        all_docs = web_docs + pdf_docs
        print(f"Total documents: {len(all_docs)}")
        
        # Overall statistics
        all_types = {}
        for doc in all_docs:
            doc_type = doc.metadata.get('doc_type', 'unknown')
            all_types[doc_type] = all_types.get(doc_type, 0) + 1
        
        print(f"\n📊 Overall Document Type Distribution:")
        for doc_type, count in sorted(all_types.items()):
            percentage = (count / len(all_docs)) * 100
            print(f"   - {doc_type}: {count} chunks ({percentage:.1f}%)")
        
        # Metadata coverage
        docs_with_dates = sum(1 for d in all_docs if 'dates' in d.metadata)
        docs_with_emails = sum(1 for d in all_docs if 'emails' in d.metadata)
        docs_with_phones = sum(1 for d in all_docs if 'phones' in d.metadata)
        
        print(f"\n📋 Metadata Extraction Coverage:")
        print(f"   - Documents with dates: {docs_with_dates}")
        print(f"   - Documents with emails: {docs_with_emails}")
        print(f"   - Documents with phones: {docs_with_phones}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n" + "="*80)
    print("✨ Testing Complete!")
    print("="*80)

if __name__ == "__main__":
    test_improvements()
