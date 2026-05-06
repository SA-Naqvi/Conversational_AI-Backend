"""
Offline indexing pipeline — run this once before starting the server.

Steps:
  1. Generate the 60-document medical corpus (data/documents/)
  2. Build the ChromaDB vector index (data/chroma_db/)

Usage:
  cd backend
  python scripts/build_index.py

Flags:
  --force   Rebuild the index even if it already exists.
"""
import sys
import logging
import argparse
import pathlib

# Add backend root to path so imports work from scripts/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser(description="Build RAG vector index from medical corpus.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if index exists.")
    args = parser.parse_args()

    # Step 1: Generate corpus
    logger.info("Step 1/2 — Generating medical document corpus...")
    from scripts.generate_corpus import generate_corpus
    generate_corpus()

    # Step 2: Build index
    logger.info("Step 2/2 — Building ChromaDB vector index...")
    from rag.indexer import build_index
    build_index(force_rebuild=args.force)

    logger.info("Done! Index is ready. You can now start the server.")


if __name__ == "__main__":
    main()
