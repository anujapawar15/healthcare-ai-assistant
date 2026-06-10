import os
import logging
import chromadb
from app.embeddings import get_embedding, get_embeddings
from app.llm import generate_answer
from app.config import settings

logger = logging.getLogger(__name__)

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        _collection = _client.get_or_create_collection(name="healthcare_docs")
    return _collection


def chunk_text(text: str) -> list:
    chunks = []
    start = 0
    while start < len(text):
        end = start + settings.chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - settings.chunk_overlap
    return chunks


def ingest_documents(data_dir: str = None) -> dict:
    data_dir = data_dir or settings.data_dir

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]

    if not files:
        raise ValueError("No .txt files found in the data directory")

    collection = get_collection()
    total_chunks = 0
    processed_files = []

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        logger.info(f"Ingesting file: {filename}")

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_text(text)
        embeddings = get_embeddings(chunks)

        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_chunks += len(chunks)
        processed_files.append(filename)
        logger.info(f"  -> {len(chunks)} chunks stored from {filename}")

    return {"files_processed": processed_files, "total_chunks": total_chunks}


def retrieve_context(question: str) -> list:
    collection = get_collection()

    if collection.count() == 0:
        return []

    top_k = min(settings.top_k_results, collection.count())
    question_embedding = get_embedding(question)

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "distance": float(dist)
        })

    return retrieved


def ask_question(question: str) -> dict:
    logger.info(f"Processing question via RAG: {question}")

    retrieved = retrieve_context(question)

    if not retrieved:
        return {
            "answer": "I could not find this information in the provided documents.",
            "sources": [],
            "confidence": "none"
        }

    context_parts = [f"[From {r['source']}]\n{r['text']}" for r in retrieved]
    context = "\n\n---\n\n".join(context_parts)

    answer = generate_answer(question, context)

    sources = [
        {
            "document": r["source"],
            "chunk": r["text"][:300] + "..." if len(r["text"]) > 300 else r["text"]
        }
        for r in retrieved
    ]

    avg_distance = sum(r["distance"] for r in retrieved) / len(retrieved)
    if avg_distance < 0.35:
        confidence = "high"
    elif avg_distance < 0.65:
        confidence = "medium"
    else:
        confidence = "low"

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence
    }
