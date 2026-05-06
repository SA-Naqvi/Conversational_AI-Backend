"""
Offline indexing pipeline for the RAG system.

Processes documents in data/documents/, chunks them with overlap,
computes embeddings, and stores them in ChromaDB.

Run via: python scripts/build_index.py
"""
import os
import re
import logging
import pathlib
from typing import List, Dict, Any

logger = logging.getLogger("medical_bot")

DOCS_DIR = pathlib.Path(__file__).parent.parent / "data" / "documents"
CHROMA_DIR = pathlib.Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "medical_docs"

CHUNK_SIZE = 512          # approximate characters per chunk (not tokens)
CHUNK_OVERLAP = 100       # character overlap between consecutive chunks
WORDS_PER_CHUNK = 80      # approximate word-based chunk size for alt strategy


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks by character count.
    Tries to split at sentence boundaries.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to cut at the last sentence boundary within the chunk
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind(".\n", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
    return chunks


def _load_documents(docs_dir: pathlib.Path) -> List[Dict[str, Any]]:
    """Load all .txt, .md, and .html files from the documents directory."""
    docs = []
    if not docs_dir.exists():
        logger.warning(f"Documents directory not found: {docs_dir}")
        return docs

    extensions = {".txt", ".md", ".html", ".rst"}
    for filepath in sorted(docs_dir.iterdir()):
        if filepath.suffix.lower() not in extensions:
            continue
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            if content.strip():
                docs.append({
                    "filename": filepath.name,
                    "filepath": str(filepath),
                    "content": content,
                })
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    logger.info(f"Loaded {len(docs)} documents from {docs_dir}")
    return docs


def build_index(force_rebuild: bool = False) -> None:
    """
    Build the ChromaDB vector index from documents.
    If the collection already exists and force_rebuild is False, skip.
    """
    import chromadb
    from rag.embeddings import embed_texts

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Check if collection already exists
    existing_collections = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_collections and not force_rebuild:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
        if count > 0:
            logger.info(f"Index already exists with {count} chunks. Skipping rebuild.")
            return

    # Delete existing collection if rebuilding
    if COLLECTION_NAME in existing_collections:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Load and chunk documents
    documents = _load_documents(DOCS_DIR)
    if not documents:
        logger.error("No documents found. Run generate_corpus.py first.")
        return

    all_ids = []
    all_texts = []
    all_metadatas = []

    for doc in documents:
        chunks = _chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['filename']}::chunk_{i}"
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metadatas.append({
                "source": doc["filename"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

    logger.info(f"Indexing {len(all_texts)} chunks from {len(documents)} documents...")

    # Compute embeddings in batches and add to collection
    BATCH = 64
    for start in range(0, len(all_texts), BATCH):
        batch_texts = all_texts[start : start + BATCH]
        batch_ids = all_ids[start : start + BATCH]
        batch_metas = all_metadatas[start : start + BATCH]

        embeddings = embed_texts(batch_texts).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=batch_metas,
        )
        logger.info(f"  Indexed {min(start + BATCH, len(all_texts))}/{len(all_texts)} chunks")

    logger.info(f"Index built successfully: {len(all_texts)} chunks in '{COLLECTION_NAME}'")


def get_client_and_collection():
    """Return a ChromaDB client and the medical docs collection."""
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return client, collection
