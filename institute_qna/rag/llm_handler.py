"""LLM Handler Module for Institute Q&A System.

This module handles interactions with Language Models (Azure OpenAI, Google Gemini)
for generating answers based on retrieved context.
"""

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai
from typing import Dict, Optional
import logging
import os

logger = logging.getLogger(__name__)


class LLMHandler:
    """Handles LLM-based answer generation."""
    
    # Default system prompt for Q&A
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant for COEP Technological University's admissions office. 
Your role is to answer questions about admissions based on the provided context in user friendly words.

Instructions:
1. Answer the question based ONLY on the provided context
2. If the answer is not in the context, try to give information on your own if related to college admission.
3. If not then say "I don't have enough information to answer that question."
4. Be concise, accurate, and professional
5. If relevant, mention the source of information
6. Format your response clearly with proper structure. 
7. For process related questions, provide step-by-step instructions.
8. Format answer properly not just provide raw text.
9. Answers should be in easy to understand language for a high school student seeking admission.
10. Use bullet points or numbered lists for clarity when needed.
11. Always maintain a polite and helpful tone.


Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(
        self, 
        provider: str = "google",
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 5000,
        system_prompt: Optional[str] = None
    ):
        """Initialize the LLM Handler.
        
        Args:
            provider: LLM provider ("azure" or "google")
            model: Model name (defaults based on provider)
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            system_prompt: Custom system prompt template
        """
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        
        # Initialize LLM based on provider
        if self.provider == "azure":
            self.model_name = model or "gpt-4o-mini"
            self.llm = self._init_azure_llm()
            self.prompt_template = ChatPromptTemplate.from_template(self.system_prompt)
            self.chain = self.prompt_template | self.llm | StrOutputParser()
        elif self.provider == "google":
            self.model_name = model or "gemini-2.5-flash"
            self.llm = self._init_google_llm()
            self.prompt_template = None
            self.chain = None
        else:
            raise ValueError(f"Unsupported provider: {provider}. Use 'azure' or 'google'")
        
        logger.info(f"Initialized LLM Handler with {self.provider} provider (model: {self.model_name})")
    
    def _init_azure_llm(self):
        """Initialize Azure OpenAI LLM."""
        try:
            return AzureChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            raise
    
    def _init_google_llm(self):
        """Initialize Google Gemini client via google.genai (no LangChain)."""
        try:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set")

            genai.configure(api_key=api_key)

            return genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                },
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini: {e}")
            raise
    
    def generate_answer(
        self, 
        question: str, 
        context: str,
        stream: bool = False
    ) -> str:
        """Generate an answer based on context.
        
        Args:
            question: User's question
            context: Retrieved context from documents
            stream: Whether to stream the response
            
        Returns:
            Generated answer string
        """
        try:
            if self.provider == "google":
                return self._generate_google_answer(question, context, stream=stream)

            # Azure path (LangChain)
            if stream:
                answer_chunks = []
                for chunk in self.chain.stream({"question": question, "context": context}):
                    answer_chunks.append(chunk)
                    print(chunk, end="", flush=True)
                return "".join(answer_chunks)
            else:
                answer = self.chain.invoke({"question": question, "context": context})
                logger.info(f"Generated answer for question: {question[:50]}...")
                return answer
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    def _generate_google_answer(self, question: str, context: str, stream: bool = False) -> str:
        """Generate answer using google.genai directly (non-LangChain)."""
        prompt = self.system_prompt.format(context=context, question=question)

        if stream:
            # Streaming responses
            answer_chunks = []
            try:
                response_stream = self.llm.generate_content(prompt, stream=True)
                for chunk in response_stream:
                    if chunk.text:
                        answer_chunks.append(chunk.text)
                        print(chunk.text, end="", flush=True)
                return "".join(answer_chunks)
            except Exception as exc:
                logger.error(f"Google Gemini streaming error: {exc}")
                raise

        # Non-streaming path
        try:
            response = self.llm.generate_content(prompt)
            # response.text aggregates candidate text; fallback to first candidate
            if hasattr(response, "text") and response.text:
                answer = response.text
            elif getattr(response, "candidates", None):
                answer = response.candidates[0].content.parts[0].text
            else:
                answer = "I couldn't generate a response."

            logger.info(f"Generated answer for question (google): {question[:50]}...")
            return answer
        except Exception as exc:
            logger.error(f"Google Gemini error: {exc}")
            raise
    
    def generate_with_metadata(
        self, 
        question: str, 
        context: str
    ) -> Dict:
        """Generate answer with metadata.
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            answer = self.generate_answer(question, context)
            
            return {
                "question": question,
                "answer": answer,
                "model": self.model_name,
                "provider": self.provider,
                "temperature": self.temperature
            }
        
        except Exception as e:
            logger.error(f"Error generating answer with metadata: {e}")
            raise
    
    def update_system_prompt(self, new_prompt: str):
        """Update the system prompt template.
        
        Args:
            new_prompt: New prompt template (must include {context} and {question})
        """
        if "{context}" not in new_prompt or "{question}" not in new_prompt:
            raise ValueError("Prompt must contain {context} and {question} placeholders")
        
        self.system_prompt = new_prompt
        if self.provider == "azure":
            self.prompt_template = ChatPromptTemplate.from_template(new_prompt)
            self.chain = self.prompt_template | self.llm | StrOutputParser()
        
        logger.info("Updated system prompt")


# Example usage
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize LLM handler
    llm_handler = LLMHandler(provider="google", temperature=0.2)
    
    # Test question and context
    question = "What are the admission requirements for B.Tech?"
    context = """
    [Document 1]
    Source: https://www.coeptech.ac.in/admissions/undergraduate/
    Content: COEP Technological University offers undergraduate B.Tech programs in various engineering disciplines. 
    Admissions are based on JEE Main scores and MHT-CET scores for Maharashtra students.
    
    [Document 2]
    Source: Admissions Brochure
    Content: Candidates must have passed 10+2 with Physics, Chemistry, and Mathematics with at least 50% marks.
    """
    
    # Generate answer
    print(f"Question: {question}\n")
    print("Generating answer...\n")
    answer = llm_handler.generate_answer(question, context)
    print(f"\nAnswer: {answer}")
