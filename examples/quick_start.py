#!/usr/bin/env python3
"""
Quick Start Script for RAG System
Run this to test your RAG system immediately!
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def check_environment():
    """Check if environment is set up correctly"""
    print("🔍 Checking environment setup...")
    
    issues = []
    
    # Check .env file
    env_file = Path(".env")
    if not env_file.exists():
        issues.append("❌ .env file not found")
    else:
        print("✅ .env file found")
    
    # Check vector database
    db_path = Path("vector_store/ug_admission_data/chroma.sqlite3")
    if not db_path.exists():
        issues.append("❌ Chroma database not found at vector_store/ug_admission_data/")
    else:
        print("✅ Chroma database found")
    
    # Check API keys
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("AZURE_OPENAI_API_KEY"):
        issues.append("⚠️  AZURE_OPENAI_API_KEY not set (required for Azure)")
    else:
        print("✅ Azure OpenAI API key found")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  {issue}")
        return False
    
    print("\n✅ All checks passed!")
    return True


def quick_test():
    """Run a quick test query"""
    print("\n" + "="*80)
    print("🚀 QUICK RAG SYSTEM TEST")
    print("="*80)
    
    try:
        from institute_qna.rag import RAGPipeline
        
        print("\n📊 Initializing RAG Pipeline...")
        pipeline = RAGPipeline(
            llm_provider="azure",
            top_k=3,
            temperature=0.2
        )
        print("✅ Pipeline initialized successfully")
        
        # Test query
        test_question = "What are the admission requirements for B.Tech programs at COEP?"
        
        print(f"\n❓ Test Question:")
        print(f"   {test_question}")
        
        print("\n⏳ Processing (this may take a few seconds)...")
        response = pipeline.query(test_question, return_sources=True)
        
        print("\n" + "="*80)
        print("📝 ANSWER:")
        print("="*80)
        print(response['answer'])
        
        print(f"\n⏱️  Processing Time: {response['processing_time']} seconds")
        print(f"📚 Sources Used: {response.get('num_sources', 0)}")
        
        if response.get('sources'):
            print("\n📖 Source Preview:")
            for i, source in enumerate(response['sources'][:2], 1):
                print(f"\n  Source {i}:")
                print(f"  {source['content'][:150]}...")
        
        print("\n" + "="*80)
        print("✅ RAG SYSTEM IS WORKING!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def interactive_mode():
    """Run interactive Q&A session"""
    print("\n" + "="*80)
    print("💬 INTERACTIVE Q&A MODE")
    print("="*80)
    print("Ask questions about COEP admissions. Type 'quit' or 'exit' to stop.")
    print("-"*80)
    
    try:
        from institute_qna.rag import RAGPipeline
        
        pipeline = RAGPipeline(
            llm_provider="azure",
            top_k=5,
            temperature=0.3
        )
        
        while True:
            print("\n❓ Your Question:")
            question = input("> ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not question:
                print("⚠️  Please enter a question")
                continue
            
            print("\n⏳ Thinking...")
            response = pipeline.query(question, return_sources=False)
            
            print("\n📝 Answer:")
            print("-"*80)
            print(response['answer'])
            print("-"*80)
            print(f"⏱️  Response time: {response['processing_time']}s")
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║            COEP INSTITUTE Q&A - RAG SYSTEM                   ║
║                   Quick Start Script                         ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Check environment
    if not check_environment():
        print("\n⚠️  Please fix the issues above before proceeding.")
        print("\nSetup instructions:")
        print("  1. Create .env file with your API keys")
        print("  2. Ensure vector database exists in vector_store/ug_admission_data/")
        print("  3. Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Menu
    print("\nChoose an option:")
    print("  1. Quick Test (single query)")
    print("  2. Interactive Mode (ask multiple questions)")
    print("  3. Exit")
    
    choice = input("\n> ").strip()
    
    if choice == "1":
        success = quick_test()
        if success:
            print("\n💡 Tip: Try interactive mode for more queries!")
            print("   Run: python examples/quick_start.py")
    elif choice == "2":
        interactive_mode()
    else:
        print("👋 Goodbye!")


if __name__ == "__main__":
    main()
