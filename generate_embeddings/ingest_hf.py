"""Build a Chroma vector store from local PDFs using HuggingFace SentenceTransformers embeddings.

- No argparse: adjust paths and flags inside main().
- Persists to ./chroma_db with collection name 'langchain' by default.
"""
# This script automates PDF loading, chunking, embedding, and storage
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

try:
    import tiktoken
except Exception:
    tiktoken = None  # optional; will fallback to char-based chunking

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise SystemExit(
        "Missing dependency: sentence-transformers. Install it with\n\n    pip install sentence-transformers\n"
    ) from e


# Defaults
DEFAULT_EMBED_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_COLLECTION_NAME = "networking_context"
DEFAULT_PERSIST_DIR = Path(__file__).parent.parent / "chroma_db"
DEFAULT_SOURCE_DIR = Path(__file__).parent.parent / "pdfs"
TOKEN_STRIDE = 200
CHUNK_SIZE = 800
BATCH_SIZE = 64

# Creates a local ChromaDB folder (chroma_db)
# purpose: So the bot can store embeddings permanently.
def init_chroma(persist_dir: Path) -> chromadb.PersistentClient:
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir), settings=Settings(anonymized_telemetry=False)
    )


def get_or_create_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
    reset: bool = False,
) -> chromadb.Collection:
    if reset:
        try:
            client.delete_collection(collection_name)
        except chromadb.errors.NotFoundError:
            pass
    return client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

# Reads each PDF page using PdfReader
# Extracts text from each page
# Tokenizes text (using tiktoken if available)
# Breaks pages into chunks of size 800 tokens
# Creates a unique ID for each chunk
def iter_pdf_text_chunks(source_dir: Path) -> Iterable[Tuple[str, str, dict]]:
    pdf_paths = sorted(source_dir.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDF files found in {source_dir}.")

    if tiktoken is not None:
        encoding = tiktoken.get_encoding("cl100k_base")
        def tokenize(text: str) -> List[int]:
            return encoding.encode(text)
        def detokenize(tokens: List[int]) -> str:
            return encoding.decode(tokens)
        step = max(CHUNK_SIZE - TOKEN_STRIDE, 1)
    else:
        # Fallback: approximate tokens via characters
        def tokenize(text: str) -> List[str]:
            return list(text)
        def detokenize(tokens: List[str]) -> str:
            return "".join(tokens)
        step = max(CHUNK_SIZE - TOKEN_STRIDE, 1)

    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path), strict=False)
        for page_number, page in enumerate(reader.pages, start=1):
            raw_text = page.extract_text() or ""
            if not raw_text.strip():
                continue
            tokens = tokenize(raw_text)
            for start in range(0, len(tokens), step):
                chunk_tokens = tokens[start : start + CHUNK_SIZE]
                chunk_text = detokenize(chunk_tokens)
                chunk_id = f"{pdf_path.stem}_p{page_number}_t{start}"
                metadata = {
                    "source": str(pdf_path.name),
                    "page": page_number,
                    "token_start": start,
                }
                yield chunk_id, chunk_text.strip(), metadata


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

#Uses HuggingFace SentenceTransformer model ###msmarco-distilbert-base-v4###
#Converts each chunk into a numerical vector
#Normalizes embeddings for cosine similarity
#purpose: Converts human text into machine-understandable vectors for semantic search.
def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    vectors = model.encode(texts, convert_to_numpy=False, normalize_embeddings=True)
    return [vec.tolist() for vec in vectors]


def main() -> None:
    source_dir = DEFAULT_SOURCE_DIR
    persist_dir = DEFAULT_PERSIST_DIR
    collection_name = DEFAULT_COLLECTION_NAME
    embed_model_name = DEFAULT_EMBED_MODEL
    reset_collection = False  # set True to rebuild

    model = SentenceTransformer(embed_model_name, device="cpu")
    chroma_client = init_chroma(persist_dir)
    collection = get_or_create_collection(chroma_client, collection_name, reset=reset_collection)

    docs_iter = iter_pdf_text_chunks(source_dir)

    for batch in batched(docs_iter, BATCH_SIZE):
        ids = [chunk_id for chunk_id, _, _ in batch]
        texts = [text for _, text, _ in batch]
        metadatas = [meta for _, _, meta in batch]
        embeddings = embed_texts(model, texts)
        collection.upsert(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    print(
        f"Ingestion complete. Persisted collection '{collection_name}' at {persist_dir.resolve()}"
    )


if __name__ == "__main__":
    main()

