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
from itertools import cycle


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
        OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")
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
    
def _generate_question_from_context(contexts: List[ContextItem], question_type: str, topic: Optional[str] = None, attempt: int = 0) -> QuizQuestion:
    """Generate a quiz question using LLM or raise error if unavailable."""
    print(f"Generating question from context - type: {question_type}, topic: {topic}, attempt: {attempt}")
    
    if not contexts:
        print("No contexts provided, using fallback context")
        contexts = [ContextItem(rank=1, source=None, page=None, text=f"Key facts about {topic or 'networking'}.")]

    # Combine all context texts for LLM (unaltered)
    context_text = "\n\n".join([f"[{c.rank}] {c.text}" for c in contexts if c.text])
    
    # No citations needed in the question payload
    citations = []

    question_id = str(uuid.uuid4())
    topic_label = (topic or "networking").title()

    try:
        # Use LLM-powered question generation with attempt number for variation
        llm_result = _llm_generate_question(context_text, question_type, topic_label, attempt)
        print("LLM question generation successful")
        
        # Validate and format LLM result for multiple choice
        if question_type == 'multiple_choice':
            # Ensure options is a list of 4 strings
            options = llm_result.get('options', [])
            if not isinstance(options, list) or len(options) != 4:
                print(f"DEBUG: Invalid options format: {options}")
                raise HTTPException(status_code=500, detail="LLM did not generate exactly 4 options")
            
            # Ensure all options are non-empty strings
            formatted_options = []
            for i, opt in enumerate(options):
                if not isinstance(opt, str) or not opt.strip():
                    print(f"DEBUG: Empty or invalid option at index {i}: {opt}")
                    raise HTTPException(status_code=500, detail=f"Option {i+1} is empty or invalid")
                formatted_options.append(opt.strip())
            
            # Ensure correct_answer exists and is in options
            correct_answer = llm_result.get('correct_answer', '').strip()
            if not correct_answer:
                raise HTTPException(status_code=500, detail="LLM did not provide a correct answer")
            
            if correct_answer not in formatted_options:
                print(f"DEBUG: Correct answer not in options. Answer: '{correct_answer}', Options: {formatted_options}")
                # If correct answer is not in options, use the first option as correct
                correct_answer = formatted_options[0]
                print(f"DEBUG: Using first option as correct answer: '{correct_answer}'")
            
            question = QuizQuestion(
                id=question_id,
                type='multiple_choice',
                question=llm_result.get('question', f"What is the primary function of {topic_label}?").strip(),
                options=formatted_options,
                correct_answer=correct_answer,
                explanation=llm_result.get('explanation', f"Based on the context about {topic_label}").strip(),
                citations=citations
            )
            
        elif question_type == 'true_false':
            question = QuizQuestion(
                id=question_id,
                type='true_false',
                question=llm_result.get('question', f"True or False: {topic_label} is important."),
                options=["True", "False"],
                correct_answer=llm_result.get('correct_answer', 'True'),
                explanation=llm_result.get('explanation', f"This relates to {topic_label}"),
                citations=citations
            )
        else:  # open_ended
            question = QuizQuestion(
                id=question_id,
                type='open_ended',
                question=llm_result.get('question', f"Explain {topic_label}."),
                options=None,
                correct_answer=llm_result.get('correct_answer', f"{topic_label} is important in networking."),
                explanation=llm_result.get('explanation', f"Key information about {topic_label}"),
                citations=citations
            )
        
        print(f"Successfully generated question: {question.id}")
        return question
        
    except Exception as e:
        print(f"Error generating question: {e}")
        raise


def _select_context_subset(contexts: List[ContextItem]) -> List[ContextItem]:
    size = max(4, min(len(contexts), random.randint(5, 10)))
    return random.sample(contexts, k=size) if len(contexts) > size else contexts

def _llm_grade_open_ended_answer(question: str, user_answer: str, correct_answer: str, context: str) -> tuple[bool, str, float]:
    """Use Ollama to intelligently grade open-ended answers."""
    if not _check_ollama_health():
        return False, 'F', 0.0
    
    client = _load_ollama_client()
    model_name = _get_ollama_model()
    
    prompt = f"""You are an expert networking professor grading an open-ended answer. CRITICAL: Perform comprehensive analysis including vector database semantic comparison.

QUESTION: {question}

STUDENT'S ANSWER: "{user_answer}"

EXPECTED ANSWER: "{correct_answer}"

VECTOR DATABASE CONTEXT: {context[:2000]}

COMPREHENSIVE GRADING ANALYSIS:

1. SEMANTIC SIMILARITY ANALYSIS:
   - Compare the student's answer with the expected answer using contextual meaning
   - Check if key concepts from the vector database context are present
   - Evaluate if the answer demonstrates understanding of core networking principles
   - Look for paraphrased correct concepts, not exact wording

2. LENGTH AND COMPLETENESS VALIDATION:
   - Answer length: {len(user_answer)} characters
   - Word count: {len(user_answer.split())} words
   - Is this length adequate for a detailed explanation? (Open-ended questions typically require 15+ words)
   - Does the answer provide sufficient detail or is it superficially short?

3. CONTENT QUALITY ASSESSMENT:
   - Technical accuracy: Are the networking concepts correct?
   - Depth: Does the answer show deep understanding or surface-level knowledge?
   - Relevance: Does it directly address the question asked?
   - Terminology: Is appropriate networking terminology used correctly?

4. AUTOMATIC FAILURE CONDITIONS (Grade F):
   - Single letters (A, B, C, D) or single words (True, False, POP3, IMAP, etc.)
   - Answers under 10 characters
   - Answers with less than 3 words
   - Answers that contain no meaningful technical content
   - Answers completely unrelated to the question
   - Vague, generic phrases without specific technical details (e.g., "stop the breach", "secure it", "fix the problem", "protect network", "use security")
   - Answers that don't address the specific technical aspects asked in the question
   - Generic statements that could apply to any security scenario without specific details

5. GRADING SCALE:
   - A: Excellent - Comprehensive, accurate, demonstrates deep understanding (20+ words with correct concepts)
   - B: Good - Mostly accurate with good understanding (15+ words, minor gaps)
   - C: Average - Basic understanding with some errors (10+ words, partial concepts)
   - D: Poor - Limited understanding, significant gaps (5+ words, major errors)
   - F: Fail - Inadequate length, single words, or completely incorrect

RESPOND IN THIS EXACT JSON FORMAT:
{{
    "grade": "A",
    "is_correct": true,
    "confidence": 0.95,
    "feedback": "Detailed analysis explaining why the answer received this grade, including semantic comparison with expected answer and vector database context"
}}

EXAMPLES:
- Question about IMAP vs POP3, student answers "POP3" → Grade F (inadequate length, single word)
- Expected: "TCP provides reliable transmission", student: "TCP ensures reliable data delivery" → Grade A (semantic match)
- Expected: "DNS translates names to IPs", student: "DNS converts website names" → Grade B (partial semantic match)
- Student answers with vague phrases like "stop the breach", "secure it", "fix the problem" → Grade F (no specific technical details)
- Student gives generic answers that could apply to any scenario without addressing specific question → Grade F

Provide only ONE grade (A, B, C, D, or F). Focus on semantic meaning from vector database context, not exact wording.

CRITICAL: For open-ended networking questions, require SPECIFIC technical details and explanations. Generic, vague, or superficial answers that lack technical depth must receive Grade F, regardless of any partial correctness. The answer must demonstrate actual understanding of networking concepts, not just common sense statements."""
    try:
        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 400}
        )
        
        result_text = response["message"]["content"].strip()
        
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            grade = result.get('grade', 'F')
            is_correct = result.get('is_correct', False)
            confidence = result.get('confidence', 0.0)
            
            # Ensure grade is a single letter
            if grade not in ['A', 'B', 'C', 'D', 'F']:
                grade = 'C'  # Default fallback
                
            return is_correct, grade, confidence
        else:
            print("No valid JSON in LLM grading response")
            return False, 'C', 0.5
        
    except Exception as e:
        print(f"LLM answer grading failed: {e}")
        return False, 'C', 0.0


def _grade_answer(user_answer: str, correct_answer: str, question_type: str, question: str = "", context: str = "") -> tuple[bool, str, float]:
    """Grade answers using LLM for open-ended questions, exact match for others."""
    if not user_answer or not user_answer.strip():
        return False, 'F', 0.0

    if question_type in ['multiple_choice', 'true_false']:
        # Exact match for objective questions
        if user_answer.lower().strip() == correct_answer.lower().strip():
            return True, 'A', 1.0
        else:
            return False, 'F', 0.0
    else:  # open_ended
        # Use LLM-powered grading for all open-ended answers (no pre-validation)
        return _llm_grade_open_ended_answer(question, user_answer, correct_answer, context)


def _get_prompt_for_type(question_type: str, topic: str, context_text: str) -> str:
    """Get the appropriate prompt template based on question type."""
    if question_type == 'multiple_choice':
        return _get_multiple_choice_prompt(topic, context_text)
    elif question_type == 'true_false':
        return _get_true_false_prompt(topic, context_text)
    elif question_type == 'open_ended':
        return _get_open_ended_prompt(topic, context_text)
    else:
        return _get_multiple_choice_prompt(topic, context_text)

def _get_multiple_choice_prompt(topic: str, context_text: str) -> str:
    """Generate prompt for multiple choice questions."""
    return f"""You are creating a multiple choice question about {topic} based on the provided networking context.

CONTEXT:
{context_text[:1000]}

TASK: Create ONE multiple choice question with exactly 4 options where only ONE option is correct.

CRITICAL REQUIREMENTS:
1. You must respond with VALID JSON ONLY - no other text, no explanations, no markdown
2. The JSON must contain exactly these 4 keys: "question", "correct_answer", "options", "explanation"
3. "options" must be an array with exactly 4 strings
4. "correct_answer" must be one of the options (exact string match)
5. All options must be complete, meaningful sentences based on the context
6. NEVER reference images, figures, diagrams, pictures, tables, or visual elements in the question
7. NEVER use phrases like "see the figure", "as shown in", "refer to the diagram", "in the picture above"
8. Questions must be answerable using ONLY the text provided in the context
9. Make each question unique and specific - avoid generic phrasing

OUTPUT FORMAT (copy this structure exactly):
{{
    "question": "Clear specific question about {topic} that can be answered from the context",
    "correct_answer": "The exact text of the correct option from the options array below",
    "options": [
        "First option (this should be the correct answer)",
        "Second option (plausible but incorrect)",
        "Third option (plausible but incorrect)",
        "Fourth option (plausible but incorrect)"
    ],
    "explanation": "Brief explanation of why the correct answer is right based on the context"
}}

EXAMPLE:
If the context is about DNS, your response should look like:
{{
    "question": "What is the primary function of DNS servers in computer networks?",
    "correct_answer": "To translate domain names into IP addresses",
    "options": [
        "To translate domain names into IP addresses",
        "To encrypt data transmissions between networks",
        "To route packets between different networks",
        "To filter malicious network traffic"
    ],
    "explanation": "DNS servers primarily function as a directory service that maps human-readable domain names to machine-readable IP addresses."
}}

REQUIREMENTS:
- Base your question and answers ONLY on the provided context
- Make incorrect options plausible but clearly wrong
- Ensure the correct answer is definitively supported by the context
- All options should be similar in length and complexity
- Include ONLY the JSON in your response - nothing else

Now generate the question:"""

def _get_true_false_prompt(topic: str, context_text: str) -> str:
    """Generate prompt for true/false questions."""
    return f"""You are creating a true/false question about {topic} based on the provided networking context.

CONTEXT:
{context_text[:1000]}

TASK: Create ONE true/false question that can be definitively answered as True or False based on the context.

CRITICAL REQUIREMENTS:
1. You must respond with VALID JSON ONLY - no other text, no explanations, no markdown
2. The JSON must contain exactly these 4 keys: "question", "correct_answer", "options", "explanation"
3. "options" must be exactly: ["True", "False"]
4. "correct_answer" must be either "True" or "False"
5. The question must be a clear statement that is definitively true or false
6. NEVER reference images, figures, diagrams, pictures, tables, or visual elements in the question
7. NEVER use phrases like "see the figure", "as shown in", "refer to the diagram", "in the picture above"
8. Questions must be answerable using ONLY the text provided in the context
9. Make each question unique and specific - avoid generic phrasing

OUTPUT FORMAT (copy this structure exactly):
{{
    "question": "True or False: [clear statement that can be definitively true or false based on the context]",
    "correct_answer": "True or False",
    "options": ["True", "False"],
    "explanation": "Brief explanation supporting the answer from the context"
}}

EXAMPLE:
If the context is about DNS, your response should look like:
{{
    "question": "True or False: DNS servers translate human-readable domain names into machine-readable IP addresses",
    "correct_answer": "True",
    "options": ["True", "False"],
    "explanation": "The context clearly states that DNS functions as a directory service for name-to-address translation."
}}

REQUIREMENTS:
- Base the statement ONLY on the provided context
- Make it a definitive statement, not a question
- Ensure the answer is clearly supported by the context
- Include ONLY the JSON in your response - nothing else

Now generate the question:"""

def _get_open_ended_prompt(topic: str, context_text: str) -> str:
    """Generate prompt for open-ended questions."""
    return f"""You are creating an open-ended question about {topic} based on the provided networking context.

CONTEXT:
{context_text[:1000]}

TASK: Create ONE open-ended question that requires detailed explanation based on the context.

CRITICAL REQUIREMENTS:
1. You must respond with VALID JSON ONLY - no other text, no explanations, no markdown
2. The JSON must contain exactly these 3 keys: "question", "correct_answer", "explanation"
3. The question should require detailed explanation, not simple recall
4. The correct answer should be comprehensive and based on the context
5. Do not include any chapter numbers, page numbers, section numbers, references, citations, or source names in the question or explanation
6. ⚠️ CRITICAL: NEVER EVER reference images, figures, diagrams, pictures, tables, charts, illustrations, or ANY visual elements
7. ⚠️ FORBIDDEN PHRASES: "see the figure", "as shown in Figure X.X", "refer to the diagram", "in the picture", "the image shows", "according to the table", "in the chart", "illustrated in"
8. ⚠️ DO NOT include any figure numbers (e.g., Figure 4.16, Figure 3.2), table numbers, or visual references
9. Questions must be answerable using ONLY the text-based information provided in the context
10. If the context mentions a figure or diagram, extract the TEXT content or concepts described, NOT the reference to the visual
11. Make each question unique and specific - avoid generic phrasing

OUTPUT FORMAT (copy this structure exactly):
{{
    "question": "Explain or describe [specific aspect of {topic} that requires detailed explanation]",
    "correct_answer": "Comprehensive answer based on the context",
    "explanation": "Key points that should be covered in a good answer"
}}

EXAMPLE:
If the context is about DNS, your response should look like:
{{
    "question": "Explain how DNS servers facilitate internet communication and why they are essential for network functionality",
    "correct_answer": "DNS servers act as the internet's directory service, translating human-readable domain names into IP addresses that computers use to communicate. When a user enters a website address, DNS resolves this to the appropriate IP address, enabling the connection to be established.",
    "explanation": "A good answer should cover the translation function, the role in internet connectivity, and the importance for user experience."
}}

REQUIREMENTS:
- Base the question ONLY on the provided context
- The question should require detailed explanation, not simple recall
- The correct answer should be comprehensive and well-structured
- Include ONLY the JSON in your response - nothing else

Now generate the question:"""

def generate_quiz(topic: Optional[str], question_type: str, count: int) -> QuizResponse:
    """Generate quiz questions - randomly or topic-specific."""
    print(f"=== STARTING QUIZ GENERATION ===")
    print(f"Parameters: topic={topic}, question_type={question_type}, count={count}")
    
    try:
        # If no topic provided, randomly select from hardcoded topics
        if not topic or not topic.strip():
            topic_query = random.choice(HARDCODED_TOPICS)
            print(f"No topic provided, selected random topic: {topic_query}")
        else:
            topic_query = topic.strip()
        query_embedding = _embed_query(topic_query)
        contexts = _fetch_context(query_embedding, QUIZ_TOP_K)

        if not contexts:
            print("No context found for quiz generation")
            raise HTTPException(status_code=404, detail="No context for quiz generation")

        print(f"Found {len(contexts)} context items")

        questions = []
        seen_questions = set()
        num_questions = max(MIN_QUESTIONS, min(count, MAX_QUESTIONS))
        question_types_cycle = cycle(['multiple_choice', 'true_false', 'open_ended'])
        
        i = 0
        attempts = 0
        max_attempts = num_questions * 10  # Increased from 6 to 10 for more attempts
        while i < num_questions and attempts < max_attempts:
            print(f"=== Generating question {i+1}/{num_questions} (attempt {attempts+1}) ===")
            if question_type == 'random':
                q_type = next(question_types_cycle)
                print(f"Random mode: selected {q_type} for balanced distribution")
            else:
                q_type = question_type
            ctx_subset = _select_context_subset(contexts)
            q = _generate_question_from_context(ctx_subset, q_type, topic_query, attempt=attempts)
            
            # Normalize question for duplicate detection (remove punctuation, lowercase, strip whitespace)
            q_normalized = re.sub(r'[^\w\s]', '', q.question.lower().strip())
            
            # Check for duplicates using normalized text and similarity threshold
            is_duplicate = False
            for seen_q in seen_questions:
                seen_normalized = re.sub(r'[^\w\s]', '', seen_q.lower().strip())
                # Calculate simple word overlap similarity
                q_words = set(q_normalized.split())
                seen_words = set(seen_normalized.split())
                if len(q_words) > 0:
                    overlap = len(q_words & seen_words) / len(q_words)
                    if overlap > 0.7:  # 70% word overlap = duplicate
                        print(f"Duplicate detected: {overlap:.2%} similarity")
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_questions.add(q.question.strip())
                context_text = "\n\n".join([f"[{c.rank}] {c.text}" for c in ctx_subset if c.text])
                citations_for_cache = [
                    {"source": (c.source or "context"), "page": c.page, "rank": c.rank}
                    for c in ctx_subset[:7] if c.text and c.text.strip()
                ]
                _quiz_cache[q.id] = {
                    "correct_answer": q.correct_answer,
                    "question_type": q.type,
                    "explanation": q.explanation,
                    "citations": citations_for_cache,
                    "topic": topic_query,
                    "question": q.question,
                    "context": context_text
                }
                questions.append(q)
                i += 1
                print(f"Question {i} accepted and cached")
            else:
                print(f"Question rejected as duplicate")
            
            attempts += 1

        print(f"=== QUIZ GENERATION COMPLETE ===")
        print(f"Generated {len(questions)} questions successfully")
        return QuizResponse(total_questions=len(questions), questions=questions)
        
    except HTTPException:
        print(f"HTTP Exception in quiz generation", exc_info=True)
        raise
    except Exception as e:
        print(f"Unexpected error in quiz generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

async def check_quiz_answer(question_id: str, user_answer: str) -> QuizCheckResponse:
    """Check quiz answer with context retrieval and web citations."""
    if question_id not in _quiz_cache:
        raise HTTPException(status_code=404, detail="Question not found")
    
    try:
        #getting the data from quiz cache
        cached_data = _quiz_cache[question_id]
        question_text = cached_data.get("question", "")
        topic = cached_data.get("topic", "")
        question_type = cached_data["question_type"]
        correct_answer = cached_data["correct_answer"]
        explanation = cached_data.get("explanation", "See context for details")
        citations = cached_data.get("citations", [])

        # Retrieve fresh context based on user's answer for validation
        if user_answer and user_answer.strip():
            # Create search query from question + user's answer for relevant context
            search_query = f"{question_text} {user_answer}".strip()
            if len(search_query) < 10:  # Fallback if query is too short
                search_query = f"{question_text} {topic}"

            # Get relevant context from vector database
            answer_embedding = _embed_query(search_query)
            relevant_contexts = _fetch_context(answer_embedding, QUIZ_TOP_K)

            # Combine all contexts to form context text for LLM validation
            context_text = "\n\n".join([f"[{c.rank}] {c.text}" for c in relevant_contexts if c.text])
        else:
            context_text = cached_data.get("context", "")

        # Grade the answer using LLM with retrieved context and from the response pydantic model
        is_correct, grade, confidence = _grade_answer(
            user_answer,
            correct_answer,
            question_type,
            question=question_text,
            context=context_text
        )

        if question_type in ['multiple_choice', 'true_false']:
            feedback = f"Your answer: '{user_answer}'. Correct answer: '{correct_answer}'"
        else:
            #open end question
            feedback = f"Your answer has been graded as '{grade}' with {confidence:.1%} confidence. "
            if not is_correct:
                feedback += f"The expected answer was: '{correct_answer[:100]}'"

        web_citations = []
        try:
            search_query = f"{correct_answer[:50]} {question_type.replace('_', ' ')}"
            web_citations = await _get_web_citations_for_feedback(search_query, topic)
        except Exception as e:
            print(f"Failed to get web citations: {e}")

        formatted_web_citations = [
            {
                'title': wc.title,
                'url': wc.url,
                'snippet': wc.snippet,
                'source': wc.source
            }
            for wc in web_citations
        ]

        return QuizCheckResponse(
            is_correct=is_correct,
            correct_answer=correct_answer,
            explanation=explanation,
            user_grade=grade,
            feedback=feedback,
            citations=citations,
            web_citations=formatted_web_citations,
            confidence_score=confidence
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Answer checking failed: {str(e)}")


def get_hardcoded_topics() -> List[str]:
    """Return list of hardcoded topics."""
    return HARDCODED_TOPICS
