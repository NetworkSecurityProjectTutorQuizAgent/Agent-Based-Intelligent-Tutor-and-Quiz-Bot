"""
Q&A Mode API - Handles question answering with context retrieval
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional
import os

import chromadb
from chromadb.config import Settings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from ollama import Client
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


# Constants
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_TOP_K = 6
OLLAMA_MODEL = None


# Pydantic Models
class AskRequest(BaseModel):
    question: str = Field(..., description="User question to retrieve context for")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="Number of contexts to retrieve")


class ContextItem(BaseModel):
    rank: int
    source: Optional[str] = None
    page: Optional[int] = None
    text: str


class AskResponse(BaseModel):
    question: str
    top_k: int
    results: List[ContextItem]
    snippet: str
    citations: List[dict]


# Global instances
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None
_ollama_client: Optional[Client] = None


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model


def _load_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
        try:
            _collection = client.get_collection(COLLECTION_NAME)
        except Exception as e:
            raise e
    return _collection


def _get_ollama_model() -> str:
    global OLLAMA_MODEL
    if OLLAMA_MODEL is None:
        load_dotenv()
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    return OLLAMA_MODEL


def _load_ollama_client() -> Client:
    global _ollama_client
    if _ollama_client is None:
        load_dotenv()
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        _ollama_client = Client(host=host)
    return _ollama_client

def _check_ollama_health() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        load_dotenv()
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = Client(host=host)
        model_name = _get_ollama_model()
        client.show(model_name)
        return True
    except Exception as e:
        print(f"Ollama health check failed: {e}")
        global _ollama_client
        _ollama_client = None
        return False

def _is_qa_related(question: str) -> bool:
    """Use LLM to check if the question is related to networking/security Q&A."""
    if not _check_ollama_health():
        return True  # Default to True if Ollama is unavailable
    
    try:
        client = _load_ollama_client()
        model_name = _get_ollama_model()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a classifier. Determine if the given question is related to networking, security, "
                    "protocols, systems, or technical topics. Respond with only 'YES' or 'NO'."
                ),
            },
            {
                "role": "user",
                "content": f"Is this question related to networking, security, or technical topics? Question: {question}",
            },
        ]
        response = client.chat(
            model=model_name,
            messages=messages,
            options={"num_predict": 10, "temperature": 0.1}
        )
        answer = response["message"]["content"].strip().upper()
        return "YES" in answer
    except Exception as e:
        print(f"Error checking if question is QA-related: {e}")
        return True  # Default to True on error


def _embed_query(text: str) -> List[float]:
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=False, normalize_embeddings=True)
    return vec[0].tolist()


def _fetch_context(query_embedding: List[float], top_k: int) -> List[ContextItem]:
    collection = _load_collection()
    # Fetch more results to ensure diversity across sources
    fetch_size = min(top_k * 3, 50)  # Fetch 3x more for filtering
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_size,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    # Separate results by source type
    slides_items = []
    textbook_items = []
    other_items = []
    
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source") if isinstance(meta, dict) else None
        page = meta.get("page") if isinstance(meta, dict) else None
        text = (doc or "").strip()
        item = ContextItem(rank=idx, source=source, page=page, text=text)
        
        # Categorize by source type
        if source and "slides" in source.lower():
            slides_items.append(item)
        elif source and any(book in source.lower() for book in ["networking", "security", "stallings"]):
            textbook_items.append(item)
        else:
            other_items.append(item)
    
    # Balance the results: prioritize slides, then mix with textbooks
    items: List[ContextItem] = []
    slides_count = min(len(slides_items), max(top_k // 2, 2))  # At least 2 slides or 50% of top_k
    textbook_count = top_k - slides_count
    
    # Add slides first
    items.extend(slides_items[:slides_count])
    # Fill remaining with textbooks
    items.extend(textbook_items[:textbook_count])
    # If still need more, add from other sources
    remaining = top_k - len(items)
    if remaining > 0:
        items.extend(other_items[:remaining])
    
    # Re-rank based on original order
    for new_rank, item in enumerate(items, start=1):
        item.rank = new_rank
    
    return items[:top_k]


def _summarize_with_ollama(question: str, contexts: List[ContextItem]) -> str:
    if not _check_ollama_health():
        raise HTTPException(status_code=503, detail="Ollama service is not available. Please ensure Ollama is running and the model is loaded.")
    
    client = _load_ollama_client()
    model_name = _get_ollama_model()
    context_text = "\n\n".join(
        [f"[{c.rank}] Source: {c.source or 'unknown'}{(' · p' + str(c.page)) if c.page else ''}\n{c.text}" for c in contexts]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful networking and security expert. Provide detailed, comprehensive answers using ONLY the provided context. "
                "Include relevant technical details, examples, and explanations. "
                "Cite sources inline like [1], [2]. If unsure, say you don't know."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]
    response = client.chat(
        model=model_name,
        messages=messages,
        options={"num_predict": 1000, "temperature": 0.7}
    )
    return response["message"]["content"].strip()


def ask_question(question: str, top_k: int = DEFAULT_TOP_K) -> AskResponse:
    """Main Q&A function"""
    q = question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Check if the question is related to Q&A topics using LLM
    if not _is_qa_related(q):
        # Let the LLM answer without database context
        if not _check_ollama_health():
            raise HTTPException(status_code=503, detail="Ollama service is not available.")
        
        client = _load_ollama_client()
        model_name = _get_ollama_model()
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer the user's question directly and briefly.",
            },
            {
                "role": "user",
                "content": q,
            },
        ]
        response = client.chat(
            model=model_name,
            messages=messages,
            options={"num_predict": 1000, "temperature": 0.7}
        )
        answer = response["message"]["content"].strip()
        prompt = "\n\n---\n\nFeel free to ask me anything related to networking, security, protocols, or technical topics!"
        return AskResponse(question=q, top_k=top_k, results=[], snippet=answer + prompt, citations=[])

    try:
        q_emb = _embed_query(q)
        contexts = _fetch_context(q_emb, top_k)
        answer = _summarize_with_ollama(q, contexts)

        citations = [
            {"index": c.rank, "source": c.source, "page": c.page}
            for c in contexts
        ]
        return AskResponse(question=q, top_k=top_k, results=contexts, snippet=answer, citations=citations)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
