"""
RAG retriever: embeds a query and returns the top-k most relevant document chunks.

Features:
- Singleton ChromaDB collection handle (connection reused across requests)
- LRU embedding cache for repeated queries
- Async-compatible via asyncio.to_thread
- Graceful degradation if index is empty or unavailable
"""
import asyncio
import hashlib
import logging
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional

logger = logging.getLogger("medical_bot")

TOP_K = 4                        # number of chunks to retrieve
MAX_CONTEXT_CHARS = 2000         # maximum total characters of retrieved context
_collection = None               # shared ChromaDB collection handle


def _get_collection():
    """Lazy-load and cache the ChromaDB collection."""
    global _collection
    if _collection is None:
        try:
            from rag.indexer import get_client_and_collection
            _, _collection = get_client_and_collection()
            count = _collection.count()
            logger.info(f"RAG collection loaded: {count} chunks available")
        except Exception as e:
            logger.error(f"Failed to load RAG collection: {e}")
    return _collection


@lru_cache(maxsize=256)
def _cached_embed(query_hash: str, query: str):
    """Cache embeddings for repeated queries (keyed by hash + text)."""
    from rag.embeddings import embed_query
    return embed_query(query).tolist()


def _retrieve_sync(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Synchronous retrieval. Returns list of dicts with keys:
      - text: the chunk content
      - source: document filename
      - score: cosine distance (lower is better)
      - chunk_index: position of chunk within source document
    """
    collection = _get_collection()
    if collection is None:
        return []

    try:
        count = collection.count()
        if count == 0:
            return []

        query_hash = hashlib.md5(query.encode()).hexdigest()
        embedding = _cached_embed(query_hash, query)

        results = collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "unknown"),
                "score": float(dist),
                "chunk_index": meta.get("chunk_index", 0),
            })

        return chunks

    except Exception as e:
        logger.error(f"RAG retrieval error: {e}")
        return []


async def retrieve(query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Async wrapper around synchronous ChromaDB retrieval.
    Runs in a thread pool to avoid blocking the event loop.
    """
    start = time.time()
    try:
        chunks = await asyncio.to_thread(_retrieve_sync, query, top_k)
        elapsed = (time.time() - start) * 1000
        logger.debug(f"RAG retrieval: {len(chunks)} chunks in {elapsed:.0f}ms")
        return chunks
    except Exception as e:
        logger.error(f"Async RAG retrieval error: {e}")
        return []


def format_context(chunks: List[Dict[str, Any]], max_chars: int = MAX_CONTEXT_CHARS) -> Optional[str]:
    """
    Format retrieved chunks into a context block for injection into the LLM prompt.
    Truncates total context to max_chars to fit within the context window.
    """
    if not chunks:
        return None

    lines = ["[RETRIEVED MEDICAL KNOWLEDGE]"]
    total_chars = len(lines[0])

    for i, chunk in enumerate(chunks, 1):
        header = f"\n[Source {i}: {chunk['source']}]"
        text = chunk["text"].strip()

        # Truncate individual chunk if very long
        if len(text) > max_chars // 2:
            text = text[: max_chars // 2] + "..."

        block = header + "\n" + text
        if total_chars + len(block) > max_chars:
            # Add partial block if space remains
            remaining = max_chars - total_chars - len(header) - 10
            if remaining > 100:
                lines.append(header)
                lines.append(text[:remaining] + "...")
            break

        lines.append(block)
        total_chars += len(block)

    lines.append("\n[END OF RETRIEVED KNOWLEDGE]")
    return "\n".join(lines)
