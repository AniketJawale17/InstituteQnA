# APIs for Institute QnA application

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path
import os
import uvicorn
from dotenv import load_dotenv
from institute_qna import WebBasedLoader
from institute_qna.data_preprocess.extract_pdf_text import PDFTextExtractor
from institute_qna.rag import RAGPipeline
from institute_qna.logging_config import configure_logging
import pandas as pd
from difflib import SequenceMatcher

load_dotenv()

# Configure application logging (fall back to logs/app.log)
log_file = os.getenv("LOGGER_FILE") or "logs/app.log"
configure_logging(level=os.getenv("LOG_LEVEL", "INFO"), log_file=log_file)

app = FastAPI(
    title="Institutional Question Anwering System",
    description="Question Answering system on COEP Admission data",
    version = '0.1.0',
    # contact="aniketjawale17@gmail.com"
    )

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATE_DIR = BASE_DIR / "templates"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

import logging
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    return_sources: Optional[bool] = True
    
class QueryResponse(BaseModel):
    question: str
    answer: str
    processing_time: float
    num_sources: Optional[int] = None
    sources: Optional[List[dict]] = None

class BatchQueryRequest(BaseModel):
    questions: List[str]
    top_k: Optional[int] = 5
    return_sources: Optional[bool] = False

# Initialize RAG Pipeline (lazy loading)
rag_pipeline: Optional[RAGPipeline] = None

def get_rag_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global rag_pipeline
    if rag_pipeline is None:
        try:
            rag_pipeline = RAGPipeline(
                persist_directory=os.getenv("RAG_PERSIST_DIRECTORY", os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_store/ug_admission_data")),
                collection_name=os.getenv("RAG_COLLECTION_NAME", os.getenv("CHROMA_COLLECTION_NAME", "UG_admission_data")),
                embedding_model=os.getenv("RAG_EMBEDDING_MODEL", os.getenv("AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
                llm_provider=os.getenv("LLM_PROVIDER", "google"),
                llm_model=os.getenv("RAG_LLM_MODEL") or None,
                top_k=int(os.getenv("RAG_TOP_K", "1")),
                temperature=float(os.getenv("LLM_TEMPERATURE", "0.3"))
            )
            logger.info("RAG Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {e}")
            raise HTTPException(status_code=500, detail="Failed to initialize RAG system")
    return rag_pipeline



@app.get("/")
async def system_status() -> dict:
    """Provides the system status as up"""
    logger.info("App started successfully")
    return {
        "Status": "Up and Running"
    }


@app.get("/chat", response_class=HTMLResponse)
async def chat_ui() -> HTMLResponse:
    """Serve the admissions Q&A chat interface."""
    html_path = TEMPLATE_DIR / "chat.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Chat UI assets missing. Please rebuild UI.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/Extract")

async def extract() -> dict:
    
    try:
        url = "https://www.coeptech.ac.in/admissions/undergraduate/"
        WebBasedLoader.load_html_markdown_from_url(url)
        
        logger.info("Generated admission data json in extracted text data")
        
        return {"Status": "Success"}
    except Exception as e:
        # Log full stack and return a 500-like message
        logger.exception("Unexpected error occurred while extracting data",e)
        # Re-raise so FastAPI returns a 500 response (or you can return a custom response)
        return {"Error" : str(e)}
    

@app.post("/Process")
async def process(pdf_path_folder : str) -> dict:
    try:
        # Call the PDF text extraction method
        extracted_text = []
        for pdf_file in os.listdir(pdf_path_folder):
            if pdf_file.endswith(".pdf"):
                pdf_path = os.path.join(pdf_path_folder, pdf_file)
                extracted_text += PDFTextExtractor.extract_text_from_text_pdf(pdf_path)
                logger.info("Extracted text from text-based PDF: %s", pdf_file)
        df = pd.DataFrame([docs.page_content for docs in extracted_text], columns=["Extracted_Text"])
        df.to_csv("extracted_text_data/extracted_pdf_text.csv", index=False)
        logger.info("Saved extracted text to CSV file")
        return {"Status": "Success", "Data": extracted_text}
    except Exception as e:
        logger.exception("Unexpected error occurred while processing PDF", e)
        return {"Error": str(e)}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> dict:
    """
    Query the RAG system with a question.
    
    Args:
        request: QueryRequest containing question and optional parameters
        
    Returns:
        QueryResponse with answer and sources
    """
    try:
        pipeline = get_rag_pipeline()
        print("RAG Pipeline obtained successfully", pipeline)
        # Process the query
        response = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            return_sources=request.return_sources,
            stream=False
        )
        
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        logger.info(f"Processed query: {request.question[:50]}...")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing query", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_query")
async def batch_query_endpoint(request: BatchQueryRequest) -> dict:
    """
    Process multiple questions in batch.
    
    Args:
        request: BatchQueryRequest with list of questions
        
    Returns:
        List of responses
    """
    try:
        pipeline = get_rag_pipeline()
        
        responses = pipeline.batch_query(
            questions=request.questions,
            top_k=request.top_k,
            return_sources=request.return_sources
        )
        
        logger.info(f"Processed batch of {len(request.questions)} queries")
        return {
            "total_questions": len(request.questions),
            "responses": responses
        }
        
    except Exception as e:
        logger.exception("Error processing batch query", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check() -> dict:
    """
    Health check endpoint to verify RAG system is ready.
    """
    try:
        pipeline = get_rag_pipeline()
        return {
            "status": "healthy",
            "rag_initialized": pipeline is not None,
            "message": "RAG system is ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/ui/meta")
async def ui_meta() -> dict:
    """Expose lightweight metadata for the UI banner."""
    pipeline = get_rag_pipeline()
    llm_handler = getattr(pipeline, "llm_handler", None)
    retriever = getattr(pipeline, "retriever", None)

    return {
        "provider": getattr(llm_handler, "provider", "unknown"),
        "model": getattr(llm_handler, "model_name", "unknown"),
        "top_k": getattr(retriever, "top_k", None),
    }


# Common admission-related search suggestions
ADMISSION_SUGGESTIONS = [
    "What are the eligibility criteria",
    "What documents are required for btech admissions",
    "What is the admission process for btech admissions",
    "What is the application deadline",
    "What about hostel facilities",
    "What is the cutoff score",
    "How to apply for B.Tech",
    "What subjects can I choose",
    "What is the course duration",
    "What about placement statistics",
    "How do I check my admission status",
    "What is the seat matrix",
    "Can I take admission through management quota",
    "What about counseling rounds",
    "What is the validity of JEE Main score",
    "How to fill the choice form",
    "What is the merit list",
    "What about CAP round admission",
]


@app.get("/autocomplete")
async def get_autocomplete_suggestions(query: str = "") -> dict:
    """
    Get autocomplete suggestions based on partial query.
    
    Args:
        query: Partial query string from user
        
    Returns:
        Dictionary with suggestions list
    """
    if not query or len(query) < 2:
        return {"suggestions": []}
    
    query_lower = query.lower().strip()

    def score_suggestion(s: str) -> float:
        s_lower = s.lower()
        ratio = SequenceMatcher(None, query_lower, s_lower).ratio()
        if s_lower.startswith(query_lower):
            ratio += 0.2
        elif query_lower in s_lower:
            ratio += 0.1
        return ratio

    scored = [(s, score_suggestion(s)) for s in ADMISSION_SUGGESTIONS]
    scored = [item for item in scored if item[1] >= 0.4]
    scored.sort(key=lambda x: (-x[1], len(x[0])))

    return {"suggestions": [s for s, _ in scored[:8]]}


if __name__ == "__main__":
    uvicorn.run("app:app", reload=False, workers=4, port=8005, host="0.0.0.0")
