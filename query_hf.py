"""Query script using HuggingFace SentenceTransformers embeddings with a Chroma DB.

- Expects a Chroma persistent directory (default: ./chroma_db) built with the same HF model.
- Defaults align with HF pipelines (collection name 'networking_context').
- No OpenAI usage.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Defaults matching many HF pipelines
DEFAULT_PERSIST_DIR = Path("chroma_db")
DEFAULT_COLLECTION_NAME = "networking_context"
DEFAULT_EMBED_MODEL = "sentence-transformers/msmarco-distilbert-base-v4"
DEFAULT_TOP_K = 4


def init_chroma_collection(persist_dir: Path, collection_name: str) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
    try:
        return client.get_collection(collection_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to open Chroma collection '{collection_name}' in '{persist_dir}'. "
            "Ensure the directory exists and the collection name matches your DB."
        ) from e


def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    vec = model.encode([query], convert_to_numpy=False, normalize_embeddings=True)
    return vec[0].tolist()


def fetch_context(collection: chromadb.Collection, query_embedding: List[float], top_k: int) -> List[str]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"],
    )
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    context_blocks = []
    for idx, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "?")
        context_blocks.append(f"[{idx}] Source: {source} (page {page})\n{doc.strip()}")
    return context_blocks


def main() -> None:
    persist_dir = DEFAULT_PERSIST_DIR
    collection_name = DEFAULT_COLLECTION_NAME
    embed_model_name = DEFAULT_EMBED_MODEL
    top_k = DEFAULT_TOP_K

    try:
        model = SentenceTransformer(embed_model_name, device="cpu")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load SentenceTransformer model '{embed_model_name}'."
        ) from e

    collection = init_chroma_collection(persist_dir, collection_name)

    try:
        query = input("Enter your question: ").strip()
    except EOFError:
        print("No input provided.")
        sys.exit(1)
    if not query:
        print("A question is required to perform retrieval.")
        sys.exit(1)

    q_emb = embed_query(model, query)
    context = fetch_context(collection, q_emb, top_k)

    if context:
        print("Context used:\n")
        for block in context:
            print(f"{block}\n")
        # Simple heuristic answer (no LLM):
        print("Answer (extractive):\n")
        print(context[0])
    else:
        print("No context retrieved.\n")
        print("Answer:\nUnable to find relevant information in the knowledge base.")


if __name__ == "__main__":
    main()
