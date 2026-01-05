"""Example usage scripts for the RAG System.

This script demonstrates how to use the Institute Q&A RAG system in various ways:
1. Direct Python usage
2. API usage with curl examples
3. Batch processing
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import from institute_qna
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


def example_1_basic_query():
    """Example 1: Basic RAG query"""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic RAG Query")
    print("="*80)
    
    from institute_qna.rag import RAGPipeline
    
    # Initialize pipeline
    pipeline = RAGPipeline(
        llm_provider="azure",  # or "google"
        top_k=3,
        temperature=0.2
    )
    
    # Ask a question
    question = "What are the admission requirements for undergraduate B.Tech programs?"
    print(f"\nQuestion: {question}")
    
    response = pipeline.query(question, return_sources=True)
    
    print(f"\nAnswer:\n{response['answer']}")
    print(f"\nProcessing Time: {response['processing_time']} seconds")
    print(f"Sources Used: {response.get('num_sources', 0)}")


def example_2_multiple_queries():
    """Example 2: Multiple queries"""
    print("\n" + "="*80)
    print("EXAMPLE 2: Multiple Queries")
    print("="*80)
    
    from institute_qna.rag import RAGPipeline
    
    pipeline = RAGPipeline(llm_provider="azure", top_k=3)
    
    questions = [
        "What departments are available at COEP?",
        "How can I apply for undergraduate admissions?",
        "What is the fee structure?",
        "Tell me about hostel facilities"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n[Q{i}]: {question}")
        response = pipeline.query(question, return_sources=False)
        print(f"[A{i}]: {response['answer'][:200]}...")


def example_3_batch_processing():
    """Example 3: Batch processing"""
    print("\n" + "="*80)
    print("EXAMPLE 3: Batch Processing")
    print("="*80)
    
    from institute_qna.rag import RAGPipeline
    
    pipeline = RAGPipeline(llm_provider="azure")
    
    questions = [
        "What are the eligibility criteria for B.Tech?",
        "When does the admission process start?",
        "What entrance exams are accepted?"
    ]
    
    print(f"\nProcessing {len(questions)} questions in batch...")
    responses = pipeline.batch_query(questions, return_sources=False)
    
    for i, resp in enumerate(responses, 1):
        print(f"\n[Question {i}]: {resp['question']}")
        print(f"[Answer {i}]: {resp['answer'][:150]}...")


def example_4_custom_retrieval():
    """Example 4: Custom retrieval parameters"""
    print("\n" + "="*80)
    print("EXAMPLE 4: Custom Retrieval Parameters")
    print("="*80)
    
    from institute_qna.rag import RAGRetriever, LLMHandler
    
    # Separate retriever and LLM handler for more control
    retriever = RAGRetriever(top_k=5)
    llm_handler = LLMHandler(provider="azure", temperature=0.1)
    
    question = "What research facilities are available?"
    print(f"\nQuestion: {question}")
    
    # Retrieve with scores
    docs = retriever.retrieve_with_scores(question, top_k=5, score_threshold=0.5)
    
    print(f"\nRetrieved {len(docs)} documents with scores:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n  Doc {i} (Score: {doc['score']:.4f}):")
        print(f"  {doc['content'][:100]}...")
    
    # Generate answer
    context = retriever.get_context_string(docs)
    answer = llm_handler.generate_answer(question, context)
    print(f"\nAnswer:\n{answer}")


def example_5_streaming_response():
    """Example 5: Streaming LLM response"""
    print("\n" + "="*80)
    print("EXAMPLE 5: Streaming Response")
    print("="*80)
    
    from institute_qna.rag import RAGRetriever, LLMHandler
    
    retriever = RAGRetriever(top_k=3)
    llm_handler = LLMHandler(provider="azure")
    
    question = "Tell me about the computer science department at COEP"
    print(f"\nQuestion: {question}")
    print("\nStreaming Answer:")
    print("-" * 80)
    
    # Retrieve context
    docs = retriever.retrieve(question)
    context = retriever.get_context_string(docs)
    
    # Stream response
    answer = llm_handler.generate_answer(question, context, stream=True)
    print("\n" + "-" * 80)


def example_6_api_usage():
    """Example 6: API usage examples"""
    print("\n" + "="*80)
    print("EXAMPLE 6: API Usage (curl commands)")
    print("="*80)
    
    print("""
# 1. Start the FastAPI server:
python app.py

# 2. Check health status:
curl http://localhost:8005/health

# 3. Single query:
curl -X POST http://localhost:8005/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What are the admission requirements?",
    "top_k": 5,
    "return_sources": true
  }'

# 4. Batch query:
curl -X POST http://localhost:8005/batch_query \\
  -H "Content-Type: application/json" \\
  -d '{
    "questions": [
      "What departments are available?",
      "What is the fee structure?"
    ],
    "top_k": 3,
    "return_sources": false
  }'

# 5. Using Python requests:
import requests

response = requests.post(
    "http://localhost:8005/query",
    json={
        "question": "Tell me about COEP admission process",
        "top_k": 5,
        "return_sources": True
    }
)

result = response.json()
print(result['answer'])
    """)


def example_7_custom_prompt():
    """Example 7: Custom system prompt"""
    print("\n" + "="*80)
    print("EXAMPLE 7: Custom System Prompt")
    print("="*80)
    
    from institute_qna.rag import RAGPipeline
    
    custom_prompt = """You are a friendly admissions counselor at COEP Technological University.
Answer questions in a conversational and encouraging tone. Include helpful tips when relevant.

Context:
{context}

Student's Question: {question}

Your Response:"""
    
    pipeline = RAGPipeline(
        llm_provider="azure",
        system_prompt=custom_prompt,
        temperature=0.5  # Higher temperature for more conversational tone
    )
    
    question = "I'm interested in mechanical engineering. Can you tell me about it?"
    print(f"\nQuestion: {question}")
    
    response = pipeline.query(question, return_sources=False)
    print(f"\nAnswer:\n{response['answer']}")


def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("INSTITUTE Q&A RAG SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        ("Basic Query", example_1_basic_query),
        ("Multiple Queries", example_2_multiple_queries),
        ("Batch Processing", example_3_batch_processing),
        ("Custom Retrieval", example_4_custom_retrieval),
        ("Streaming Response", example_5_streaming_response),
        ("API Usage", example_6_api_usage),
        ("Custom Prompt", example_7_custom_prompt)
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nEnter example number to run (1-7), 'all' for all examples, or 'q' to quit:")
    choice = input("> ").strip().lower()
    
    if choice == 'q':
        return
    elif choice == 'all':
        for name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\nError in {name}: {e}")
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        try:
            examples[int(choice) - 1][1]()
        except Exception as e:
            print(f"\nError: {e}")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
