# Networking RAG System

An intelligent Retrieval-Augmented Generation (RAG) system for networking and security education. This system provides two modes: **Q&A Mode** for answering networking questions with context-based responses, and **Quiz Mode** for interactive learning with intelligent question generation and grading.

---

## Quick Setup

### Prerequisites

- **Python**: 3.12.8
- **Ollama**: Latest version
- **RAM**: 4GB+ recommended
- **Storage**: ~1GB for models and database

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .rag

# Activate virtual environment
# On Windows:
.rag\Scripts\activate

# On Linux/Mac:
source .rag/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Ollama Setup

Download and install Ollama from [ollama.com](https://ollama.com/download)

```bash
# Pull the Gemma3 1B model
ollama pull gemma3:1b

# Start Ollama service (runs automatically on first use)
ollama serve
```

### 4. Run the Application

```bash
python main.py
```

Access the web interface at: **http://localhost:8000**

---

## Technical Stack

### Embedding Model

- **Model**: SentenceTransformers (msmarco-distilbert-base-v4)
- **Purpose**: Converts text queries and documents into vector embeddings for semantic search
- **Dimension**: 768-dimensional vectors

### Language Model

- **Model**: Ollama with Gemma3:1b
- **Purpose**: Generates contextual answers and quiz questions
- **Type**: Local LLM (runs on your machine)

### Vector Database

- **Database**: ChromaDB
- **Purpose**: Stores and retrieves document embeddings for fast semantic search

### API & Frontend

- **Backend**: FastAPI
- **Frontend**: Modern HTML/CSS/JavaScript interface
- **Communication**: REST API with JSON payloads

---

## System Flow

### Q&A Mode

The Q&A mode allows users to ask questions about networking topics and receive contextual answers with citations.

**Flow Overview:**

1. **Question Input**: User enters a question in the chat interface
2. **Generic Check**: The system uses the LLM to determine if the question is generic (e.g., "hi", "how are you") or domain-specific
3. **Generic Response**: If generic, the LLM provides a friendly response and prompts the user to ask networking questions
4. **Domain-Specific Processing**:
   - The question is converted into a vector embedding
   - The embedding is used to search the vector database for relevant document chunks
   - Top-K similar chunks are retrieved (default: 6 chunks)
5. **Answer Generation**: The LLM processes the question along with the retrieved context to generate a comprehensive answer
6. **Citation Mapping**: The system identifies which source documents were used and includes them as citations
7. **Response Display**: The answer is displayed with inline citations showing document sources and page numbers

### Quiz Mode

The Quiz mode provides interactive learning through intelligent question generation and grading.

**Flow Overview:**

1. **Quiz Configuration**:

   - **Generation Type**: Choose between Random Questions or Topic-Specific Questions
   - **Topic Selection**: For topic-specific mode, either select from predefined topics or type a custom topic
   - **Question Type**: Select question format (Multiple Choice, True/False, Open-Ended, or Random Mix)
   - **Question Count**: Specify number of questions (1-10)

2. **Question Generation**:

   - The selected topic is converted into a vector embedding
   - The embedding retrieves relevant context chunks from the vector database
   - The LLM generates questions based on the retrieved context
   - Questions are formatted according to the selected type

3. **Question Presentation**:

   - Multiple Choice: User selects from 4 options
   - True/False: User selects True or False
   - Open-Ended: User types a detailed answer

4. **Answer Evaluation**:

   - **Objective Questions** (Multiple Choice, True/False): Exact match comparison
   - **Open-Ended Questions**: LLM-based grading considering accuracy, depth, and terminology
   - Confidence scores are calculated for each answer

5. **Feedback & Citations**:
   - Correct answer is displayed
   - Detailed explanation is provided
   - Database citations show relevant source material
   - Web citations provide additional references
   - User grade (A-F) and confidence percentage are shown

**Key Features:**

- Flexible topic selection (predefined or custom)
- Multiple question types for varied learning
- Intelligent LLM-based grading for open-ended questions
- Comprehensive feedback with citations
- Confidence-based scoring system

---

## Architecture Diagrams

### Q&A Mode Agent Flow

![1762106812431](image/README/1762106812431.png)

**Process Details:**

- Generic detection prevents unnecessary database queries
- Semantic search finds contextually relevant documents
- LLM synthesizes information from multiple sources
- Citations provide source transparency

### Quiz Mode Agent Flow

![1762106791641](image/README/1762106791641.png)

---
