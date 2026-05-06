"""
Singleton wrapper for the sentence-transformers embedding model.
Uses all-MiniLM-L6-v2 (22M params, 384-dim, CPU-efficient).
"""
import logging
import os
from typing import List
import numpy as np

logger = logging.getLogger("medical_bot")

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model = None


def get_embedding_model():
    """Return the singleton SentenceTransformer model, loading it on first call."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {MODEL_NAME}")
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    return _model


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a batch of texts. Returns (N, dim) numpy array."""
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns (dim,) numpy array."""
    model = get_embedding_model()
    return model.encode(
        [query],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]
