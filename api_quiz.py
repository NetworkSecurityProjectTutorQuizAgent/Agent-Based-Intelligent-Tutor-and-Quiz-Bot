"""
Quiz Mode API - Handles quiz generation and answer checking with web citations
"""
from __future__ import annotations

from pathlib import Path
import random
import uuid
import os
from typing import List, Optional, Dict
from urllib.parse import quote_plus

import aiohttp
import chromadb
from chromadb.config import Settings
from fastapi import HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from ollama import Client
import json
import re


# Constants
PERSIST_DIR = Path("chroma_db")
COLLECTION_NAME = "networking_context"
EMBED_MODEL_NAME = "sentence-transformers/msmarco-distilbert-base-v4"
OLLAMA_MODEL = None
QUIZ_TOP_K = 12
MIN_QUESTIONS = 1
MAX_QUESTIONS = 10
WEB_SEARCH_TIMEOUT = 10

# Hardcoded topics from the networking vector database
HARDCODED_TOPICS = [
    "Firewalls",
    "DNS (Domain Name System)",
    "TCP/IP Protocol",
    "Network Security",
    "Encryption and SSL/TLS",
    "VPN (Virtual Private Network)",
    "DDoS Attacks",
    "HTTP and HTTPS",
    "Network Routing",
    "OSI Model",
    "IP Addressing and Subnetting",
    "Network Authentication",
    "Wireless Security (WPA/WPA2)",
    "Intrusion Detection Systems (IDS)",
    "Load Balancing",
    "Network Protocols",
    "Packet Switching",
    "Network Topology",
    "Cybersecurity Threats",
    "Email Security (SMTP, POP3, IMAP)"
]


# Pydantic Models
class ContextItem(BaseModel):
    rank: int
    source: Optional[str] = None
    page: Optional[int] = None
    text: str


class WebCitation(BaseModel):
    title: str
    url: str
    snippet: str
    source: str


class QuizRequest(BaseModel):
    topic: Optional[str] = None
    question_type: str = 'multiple_choice'
    count: int = 1


class QuizQuestion(BaseModel):
    id: str
    type: str
    question: str
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: str
    citations: List[dict]


class QuizResponse(BaseModel):
    total_questions: int
    questions: List[QuizQuestion]


class QuizCheckRequest(BaseModel):
    question_id: str
    user_answer: str


class QuizCheckResponse(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    user_grade: str
    feedback: str
    citations: List[dict]
    web_citations: List[dict]
    confidence_score: float


# Global instances
_model: Optional[SentenceTransformer] = None
_collection: Optional[chromadb.Collection] = None
_ollama_client: Optional[Client] = None
_quiz_cache: Dict[str, Dict[str, any]] = {}


def _load_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    return _model


def _get_ollama_model() -> str:
    global OLLAMA_MODEL
    if OLLAMA_MODEL is None:
        load_dotenv()
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")
    return OLLAMA_MODEL


def _check_ollama_health() -> bool:
    """Check if Ollama is running and the model is available."""
    try:
        client = _load_ollama_client()
        model_name = _get_ollama_model()
        client.show(model_name)
        return True
    except Exception as e:
        global _ollama_client
        _ollama_client = None
        return False


def _load_ollama_client() -> Client:
    global _ollama_client
    if _ollama_client is None:
        load_dotenv()
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        _ollama_client = Client(host=host)
    return _ollama_client


def _load_collection() -> chromadb.Collection:
    """Load the chroma collection for context retrieval."""
    global _collection
    if _collection is None:
        print(f"Loading chroma collection: {COLLECTION_NAME} from {PERSIST_DIR}")
        client = chromadb.PersistentClient(path=str(PERSIST_DIR), settings=Settings(anonymized_telemetry=False))
        _collection = client.get_collection(COLLECTION_NAME)
        print("ChromaDB collection loaded successfully")
    return _collection


def _embed_query(text: str) -> List[float]:
    print(f"Generating embedding for text: {text[:100]}...")
    model = _load_model()
    vec = model.encode([text], convert_to_numpy=False, normalize_embeddings=True)
    embedding = vec[0].tolist()
    print(f"Generated embedding of length: {len(embedding)}")
    return embedding


def _fetch_context(query_embedding: List[float], top_k: int) -> List[ContextItem]:
    print(f"Fetching context from vector database, top_k={top_k}")
    collection = _load_collection()
    try:
        # Fetch more results to ensure diversity across sources
        fetch_size = min(top_k * 3, 50)  # Fetch 3x more for filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_size,
            include=["documents", "metadatas"],
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        print(f"Retrieved {len(documents)} documents from vector database")
        
        # Separate results by source type
        slides_items = []
        textbook_items = []
        other_items = []
        
        for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
            source = meta.get("source") if isinstance(meta, dict) else None
            page = meta.get("page") if isinstance(meta, dict) else None
            text = (doc or "").strip()
            item = ContextItem(rank=idx, source=source, page=page, text=text)
            print(f"Context {idx}: source={source}, page={page}, text_length={len(text)}")
            
            # Categorize by source type
            if source and "slides" in source.lower():
                slides_items.append(item)
            elif source and any(book in source.lower() for book in ["networking", "security", "stallings"]):
                textbook_items.append(item)
            else:
                other_items.append(item)
        
        print(f"Source distribution: {len(slides_items)} slides, {len(textbook_items)} textbooks, {len(other_items)} other")
        
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
        
        print(f"Final mix: {sum(1 for i in items if i.source and 'slides' in i.source.lower())} slides, {sum(1 for i in items if i.source and any(b in i.source.lower() for b in ['networking', 'security', 'stallings']))} textbooks")
        
        return items[:top_k]
    except Exception as e:
        print(f"Error fetching context from vector database: {e}")
        raise


async def _search_web(query: str, max_results: int = 3) -> List[WebCitation]:
    """Search the web for relevant information and return citations."""
    try:
        search_url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_html=1&skip_disambig=1"

        timeout = aiohttp.ClientTimeout(total=WEB_SEARCH_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(search_url) as response:
                if response.status == 200:
                    data = await response.json()
                    citations = []

                    if data.get('AbstractText') and data['AbstractText'].strip():
                        citations.append(WebCitation(
                            title=data.get('Answer', query)[:100],
                            url=data.get('AbstractURL', ''),
                            snippet=data['AbstractText'][:200],
                            source='DuckDuckGo'
                        ))

                    if data.get('RelatedTopics'):
                        for topic in data['RelatedTopics'][:max_results-1]:
                            if isinstance(topic, dict) and topic.get('Text'):
                                citations.append(WebCitation(
                                    title=topic['Text'][:100],
                                    url=topic.get('FirstURL', ''),
                                    snippet=topic['Text'][:200],
                                    source='DuckDuckGo'
                                ))

                    return citations[:max_results]
                else:
                    return []
    except Exception as e:
        print(f"Web search error: {e}")
        return []


def _llm_generate_question(context_text: str, question_type: str, topic: str, attempt: int = 0) -> Dict[str, any]:
    """Use Ollama to generate intelligent quiz questions based on context."""
    print(f"Generating question using LLM for topic: {topic}, type: {question_type}, attempt: {attempt}")
    if not _check_ollama_health():
        raise HTTPException(status_code=503, detail="Ollama service is not available. Please ensure Ollama is running and the model is loaded.")
    
    client = _load_ollama_client()
    model_name = _get_ollama_model()
    
    prompt = _get_prompt_for_type(question_type, topic, context_text)
    
    # Increase temperature based on attempt to get more variation
    temperature = min(0.7 + (attempt * 0.1), 1.0)
    
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": 500}
        )
        
        result_text = response["message"]["content"].strip()
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            print(f"DEBUG: Parsed JSON result:\n{result}")
            return result
        else:
            print("DEBUG: No JSON found in LLM response, raising error")
            raise HTTPException(status_code=500, detail="Failed to generate valid question format from LLM")
            
    except Exception as e:
        print(f"LLM question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

