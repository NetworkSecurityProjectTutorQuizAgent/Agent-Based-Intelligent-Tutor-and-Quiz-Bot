"""
Main FastAPI Application - Networking RAG System
Combines Q&A and Quiz modes with separate module imports
Now includes Security & Privacy features and Network Tracing
"""
from __future__ import annotations

from pathlib import Path
from typing import List
import uuid
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

# Import Q&A module
from api_qa import (
    AskRequest,
    AskResponse,
    ask_question,
    _load_model,
    _load_collection,
    _check_ollama_health
)

# Import Quiz module
from api_quiz import (
    QuizRequest,
    QuizResponse,
    QuizCheckRequest,
    QuizCheckResponse,
    generate_quiz,
    check_quiz_answer,
    get_hardcoded_topics
)

# Import Security & Network Trace modules
from security_middleware import SecurityMiddleware, get_security_stats
from api_network_trace import (
    get_tracer,
    explain_trace,
    TraceQueryRequest,
    TraceListResponse,
    AnalyticsResponse
)

# Check Ollama status at startup
ollama_status = _check_ollama_health()
print(f"Starting Networking RAG System...")
print(f"Loading Ollama model === {os.getenv('OLLAMA_MODEL')}")
print(f"Ollama Model Status: {'RUNNING' if ollama_status else 'NOT AVAILABLE'}")
print(f"Vector Database: LOADED")
print(f"Security Features: ENABLED")
print(f"Network Tracing: ENABLED")
print(f"Server will start on http://127.0.0.1:8000")
print("=" * 50)


# Initialize FastAPI app
app = FastAPI(
    title="Networking RAG with Security & Tracing",
    description="RAG system with built-in security safeguards and network trace analysis"
)

# Add Security Middleware FIRST (before CORS)
app.add_middleware(SecurityMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

#checks health of application
@app.get("/health")
def health():
    """Health check endpoint."""
    try:
        _load_model() #sentese Transformeners 
        _load_collection() #context
        ollama_ok = _check_ollama_health()
        return {
            "status": "ok" if ollama_ok else "warning", 
            "ollama": "ok" if ollama_ok else "not available",
            "embedding_model": "ok",
            "vector_db": "ok",
            "security": "enabled",
            "tracing": "enabled"
        }
    except Exception as e:
        return {"status": "error", "detail": str(e)}


@app.get("/")
def root_ui():
    """Serve the main UI."""
    index_path = Path("templates/index.html")
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="UI not found")
    return FileResponse(index_path)


@app.get("/quiz/topics")
def get_topics():
    """Get list of hardcoded topics for quiz generation."""
    return {"topics": get_hardcoded_topics()}


@app.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, request: Request):
    """Q&A endpoint - Answer questions with context from the database."""
    tracer = get_tracer()
    trace_id = str(uuid.uuid4())
    
    # Start trace
    tracer.start_trace(trace_id, "/ask", "POST")
    
    # Capture HTTP request packet
    request_headers = dict(request.headers)
    request_body = payload.model_dump_json()
    tracer.capture_http_request(
        trace_id,
        "POST",
        str(request.url),
        request_headers,
        request_body
    )
    
    try:
        import time
        
        # Log input processing
        start = time.time()
        tracer.add_event(
            trace_id, 'INPUT_VALIDATION', 'Security_Layer',
            metadata={'question_length': len(payload.question)}
        )
        
        # Log embedding generation
        start_embed = time.time()
        result = ask_question(payload.question, payload.top_k)
        embed_duration = (time.time() - start_embed) * 1000
        
        tracer.add_event(
            trace_id, 'EMBEDDING_GENERATION', 'SentenceTransformer',
            duration_ms=embed_duration,
            data_size_bytes=len(payload.question.encode('utf-8'))
        )
        
        # Log vector search
        tracer.add_event(
            trace_id, 'VECTOR_SEARCH', 'ChromaDB',
            metadata={'top_k': payload.top_k, 'results_found': len(result.results)}
        )
        
        # Log LLM generation
        tracer.add_event(
            trace_id, 'LLM_GENERATION', 'Ollama',
            metadata={'answer_length': len(result.snippet)}
        )
        
        # Log response
        tracer.add_event(
            trace_id, 'RESPONSE_SENT', 'API_Gateway',
            data_size_bytes=len(result.model_dump_json().encode('utf-8'))
        )
        
        # Capture HTTP response packet
        response_dict = result.model_dump()
        response_dict['trace_id'] = trace_id
        response_body = json.dumps(response_dict)
        response_headers = {
            'content-type': 'application/json',
            'content-length': str(len(response_body))
        }
        tracer.capture_http_response(trace_id, 200, response_headers, response_body)
        
        tracer.end_trace(trace_id, 'completed')
        
        return JSONResponse(content=response_dict)
        
    except Exception as e:
        tracer.add_event(trace_id, 'ERROR', 'System', metadata={'error': str(e)})
        
        # Capture error response
        error_response = {"detail": str(e)}
        tracer.capture_http_response(trace_id, 500, {'content-type': 'application/json'}, json.dumps(error_response))
        
        tracer.end_trace(trace_id, 'failed')
        raise


@app.post("/quiz/generate", response_model=QuizResponse)
async def generate_quiz_endpoint(payload: QuizRequest, request: Request):
    """Quiz generation endpoint - Generate quiz questions."""
    tracer = get_tracer()
    trace_id = str(uuid.uuid4())
    
    tracer.start_trace(trace_id, "/quiz/generate", "POST")
    
    # Capture HTTP request
    tracer.capture_http_request(
        trace_id, "POST", str(request.url),
        dict(request.headers), payload.model_dump_json()
    )
    
    try:
        tracer.add_event(
            trace_id, 'QUIZ_GENERATION_START', 'Quiz_Engine',
            metadata={'topic': payload.topic, 'type': payload.question_type, 'count': payload.count}
        )
        
        result = generate_quiz(payload.topic, payload.question_type, payload.count)
        
        tracer.add_event(
            trace_id, 'QUIZ_GENERATION_COMPLETE', 'Quiz_Engine',
            metadata={'questions_generated': len(result.questions)}
        )
        
        response_dict = result.model_dump()
        response_dict['trace_id'] = trace_id
        
        # Capture HTTP response
        tracer.capture_http_response(
            trace_id, 200,
            {'content-type': 'application/json'},
            json.dumps(response_dict)
        )
        
        tracer.end_trace(trace_id, 'completed')
        
        return JSONResponse(content=response_dict)
        
    except Exception as e:
        tracer.add_event(trace_id, 'ERROR', 'System', metadata={'error': str(e)})
        tracer.capture_http_response(trace_id, 500, {'content-type': 'application/json'}, json.dumps({"detail": str(e)}))
        tracer.end_trace(trace_id, 'failed')
        raise


@app.post("/quiz/check", response_model=QuizCheckResponse)
async def check_quiz_endpoint(payload: QuizCheckRequest, request: Request):
    """Quiz checking endpoint - Check user answers with web citations."""
    tracer = get_tracer()
    trace_id = str(uuid.uuid4())
    
    tracer.start_trace(trace_id, "/quiz/check", "POST")
    
    try:
        tracer.add_event(
            trace_id, 'ANSWER_VALIDATION_START', 'Quiz_Grader',
            metadata={'question_id': payload.question_id, 'answer_length': len(payload.user_answer)}
        )
        
        result = await check_quiz_answer(payload.question_id, payload.user_answer)
        
        tracer.add_event(
            trace_id, 'ANSWER_GRADING_COMPLETE', 'Quiz_Grader',
            metadata={'is_correct': result.is_correct, 'grade': result.user_grade}
        )
        
        tracer.end_trace(trace_id, 'completed')
        
        response_dict = result.model_dump()
        response_dict['trace_id'] = trace_id
        return JSONResponse(content=response_dict)
        
    except Exception as e:
        tracer.add_event(trace_id, 'ERROR', 'System', metadata={'error': str(e)})
        tracer.end_trace(trace_id, 'failed')
        raise


# === Security & Monitoring Endpoints ===

@app.get("/security/stats")
def security_stats():
    """Get security statistics."""
    return get_security_stats()


@app.get("/trace/recent")
def get_recent_traces(limit: int = 10):
    """Get recent network traces."""
    tracer = get_tracer()
    traces = tracer.get_recent_traces(limit)
    return TraceListResponse(traces=traces, total=len(traces))


@app.post("/trace/explain")
def explain_trace_endpoint(payload: TraceQueryRequest):
    """Get detailed explanation of a specific trace."""
    tracer = get_tracer()
    trace = tracer.get_trace(payload.trace_id)
    
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    explanation = explain_trace(trace)
    
    return {
        "trace_id": payload.trace_id,
        "trace": trace,
        "explanation": explanation
    }


@app.get("/trace/analytics")
def get_trace_analytics():
    """Get analytics across all traces."""
    tracer = get_tracer()
    analytics = tracer.get_analytics()
    return AnalyticsResponse(analytics=analytics)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

