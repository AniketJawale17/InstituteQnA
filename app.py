# APIs for Institute QnA application

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
import uvicorn
import re
import time
import threading
from copy import deepcopy
from datetime import datetime, timezone
from dotenv import load_dotenv
from institute_qna.data_preprocess.knoweldge_base_creation import main as generate_knowledge_base
from institute_qna.rag import RAGPipeline
from institute_qna.logging_config import configure_logging
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


class KnowledgeBaseGenerateRequest(BaseModel):
    extraction_method: Optional[str] = "azure"


KB_TOTAL_STEPS = 6
KB_PLANNED_URLS = int(os.getenv("KB_PLANNED_URLS", "15"))
_kb_lock = threading.Lock()
_kb_state: Dict[str, Any] = {
    "status": "idle",
    "running": False,
    "progress_percent": 0,
    "current_step": "Not started",
    "started_at": None,
    "finished_at": None,
    "extraction_method": None,
    "duration_seconds": None,
    "error": None,
    "events": [],
    "summary": {
        "planned_urls": KB_PLANNED_URLS,
        "total_documents": 0,
        "sample_document": None,
    },
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_kb_event(message: str) -> None:
    _kb_state["events"].append({"time": _utc_now_iso(), "message": message})
    _kb_state["events"] = _kb_state["events"][-200:]


def _parse_progress_percent(message: str, current_percent: int) -> int:
    percent_match = re.match(r"^\s*(\d{1,3})%\s*-", message)
    if percent_match:
        return max(current_percent, min(100, int(percent_match.group(1))))

    started_match = re.search(r"Step\s+(\d+)\s*/\s*6\s*:\s*.*started", message, re.IGNORECASE)
    if started_match:
        step = int(started_match.group(1))
        return max(current_percent, int(((step - 1) / KB_TOTAL_STEPS) * 100))

    complete_match = re.search(r"Step\s+(\d+)\s*/\s*6\s*complete", message, re.IGNORECASE)
    if complete_match:
        step = int(complete_match.group(1))
        return max(current_percent, int((step / KB_TOTAL_STEPS) * 100))

    if "completed" in message.lower() or "finished" in message.lower():
        return max(current_percent, 100)

    return current_percent


def _run_knowledge_base_job(extraction_method: str) -> None:
    start_time = time.time()

    def _progress_callback(message: str) -> None:
        with _kb_lock:
            current_percent = int(_kb_state.get("progress_percent", 0) or 0)
            _kb_state["progress_percent"] = _parse_progress_percent(message, current_percent)
            _kb_state["current_step"] = message
            _append_kb_event(message)

    try:
        docs = generate_knowledge_base(
            extraction_method=extraction_method,
            progress_callback=_progress_callback,
        )

        sample_document = None
        if docs:
            sample_document = {
                "content_preview": docs[0].page_content[:300],
                "metadata": docs[0].metadata,
            }

        duration = round(time.time() - start_time, 2)

        with _kb_lock:
            _kb_state["status"] = "completed"
            _kb_state["running"] = False
            _kb_state["progress_percent"] = 100
            _kb_state["current_step"] = "Knowledge base creation completed"
            _kb_state["finished_at"] = _utc_now_iso()
            _kb_state["duration_seconds"] = duration
            _kb_state["error"] = None
            _kb_state["summary"] = {
                "planned_urls": KB_PLANNED_URLS,
                "total_documents": len(docs),
                "sample_document": sample_document,
            }
            _append_kb_event(f"Completed in {duration} seconds with {len(docs)} documents")

        logger.info(
            "Knowledge base generation completed with method '%s' and %d documents",
            extraction_method,
            len(docs),
        )
    except Exception as e:
        duration = round(time.time() - start_time, 2)
        with _kb_lock:
            _kb_state["status"] = "failed"
            _kb_state["running"] = False
            _kb_state["finished_at"] = _utc_now_iso()
            _kb_state["duration_seconds"] = duration
            _kb_state["error"] = str(e)
            _kb_state["current_step"] = "Knowledge base creation failed"
            _append_kb_event(f"Failed: {e}")
        logger.exception("Unexpected error occurred while generating knowledge base")


def _kb_state_snapshot() -> Dict[str, Any]:
    with _kb_lock:
        return deepcopy(_kb_state)

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


@app.get("/knowledge_base/generate", response_class=HTMLResponse)
async def knowledge_base_generate_ui() -> HTMLResponse:
    """Serve knowledge base generation UI with runtime progress and summary."""
    html_path = TEMPLATE_DIR / "knowledge_base_generate.html"
    if not html_path.exists():
        raise HTTPException(status_code=500, detail="Knowledge base UI assets missing.")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/knowledge_base/generate")
async def knowledge_base_generate(request: KnowledgeBaseGenerateRequest) -> dict:
    """Start knowledge base generation in background and return current run state."""
    extraction_method = (request.extraction_method or "azure").strip().lower()
    if extraction_method not in {"azure", "opensource"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid extraction_method. Use 'azure' or 'opensource'.",
        )

    with _kb_lock:
        if _kb_state["running"]:
            raise HTTPException(
                status_code=409,
                detail="Knowledge base generation is already running.",
            )

        _kb_state["status"] = "running"
        _kb_state["running"] = True
        _kb_state["progress_percent"] = 0
        _kb_state["current_step"] = "Queued"
        _kb_state["started_at"] = _utc_now_iso()
        _kb_state["finished_at"] = None
        _kb_state["extraction_method"] = extraction_method
        _kb_state["duration_seconds"] = None
        _kb_state["error"] = None
        _kb_state["events"] = []
        _kb_state["summary"] = {
            "planned_urls": KB_PLANNED_URLS,
            "total_documents": 0,
            "sample_document": None,
        }
        _append_kb_event(
            f"Run started (method={extraction_method}, planned_urls={KB_PLANNED_URLS})"
        )

    worker = threading.Thread(
        target=_run_knowledge_base_job,
        args=(extraction_method,),
        daemon=True,
        name="knowledge-base-generation-worker",
    )
    worker.start()

    return {
        "status": "started",
        "run": _kb_state_snapshot(),
    }


@app.get("/knowledge_base/generate/status")
async def knowledge_base_generate_status() -> dict:
    """Return current knowledge base generation state for runtime UI polling."""
    return _kb_state_snapshot()


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
