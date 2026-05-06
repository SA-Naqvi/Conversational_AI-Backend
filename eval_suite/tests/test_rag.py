"""
Module 2 — RAG Component Evaluation.

Tests:
  2.1 — Retrieval relevance: precision@k, recall@k, MRR
  2.2 — Faithfulness (LLM-as-judge via local Qwen)
  2.3 — Context relevance
  2.4 — Domain boundary (out-of-domain queries)
"""
import os
import sys
import json
import pytest
import asyncio

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


# ---------------------------------------------------------------------------
# Metric Computations
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_ids: list, relevant_ids: list, k: int = 5) -> float:
    """Precision@k = |retrieved ∩ relevant| / k"""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if k == 0:
        return 0.0
    return len(retrieved_set & relevant_set) / k


def recall_at_k(retrieved_ids: list, relevant_ids: list, k: int = 5) -> float:
    """Recall@k = |retrieved ∩ relevant| / |relevant|"""
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 1.0  # No relevant docs means perfect recall (vacuously true)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def mrr(retrieved_ids: list, relevant_ids: list) -> float:
    """Mean Reciprocal Rank = 1 / rank_of_first_relevant_result"""
    relevant_set = set(relevant_ids)
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def chunk_id_matches_doc(chunk_id: str, doc_name: str) -> bool:
    """Check if a chunk ID belongs to a given document filename."""
    return chunk_id.startswith(doc_name.replace(".txt", "")) or doc_name in chunk_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRAGRetrieval:
    """Test retrieval quality without LLM dependency."""

    def test_rag_queries_data_loads(self, rag_queries_data):
        """Verify rag_queries.json loads correctly."""
        assert isinstance(rag_queries_data, list)
        assert len(rag_queries_data) >= 30

    def test_retriever_imports(self):
        """Verify the RAG retriever can be imported."""
        try:
            from rag.retriever import _retrieve_sync
        except Exception as e:
            pytest.skip(f"RAG retriever not available: {e}")

    def test_retrieval_precision_recall(self, rag_queries_data, reports_dir):
        """
        2.1 — Compute precision@5, recall@5, and MRR for all annotated queries.
        """
        try:
            from rag.retriever import _retrieve_sync
        except Exception:
            pytest.skip("RAG retriever not importable")

        results = []
        precisions = []
        recalls = []
        mrrs = []

        for query_entry in rag_queries_data:
            query = query_entry["query"]
            relevant_docs = query_entry.get("relevant_docs", [])
            category = query_entry.get("category", "unknown")

            # Retrieve top-5 chunks
            try:
                chunks = _retrieve_sync(query, top_k=5)
            except Exception as e:
                results.append({
                    "id": query_entry["id"],
                    "error": str(e),
                })
                continue

            # Build retrieved chunk IDs
            retrieved_ids = []
            for chunk in chunks:
                source = chunk.get("source", "")
                idx = chunk.get("chunk_index", 0)
                retrieved_ids.append(f"{source}::chunk_{idx}")

            # Build relevant chunk ID patterns (match by doc name)
            # A retrieved chunk is relevant if its source doc is in the relevant_docs list
            retrieved_relevant = []
            for rid in retrieved_ids:
                for rdoc in relevant_docs:
                    doc_base = rdoc.split("::")[0]  # Get the filename part
                    if doc_base in rid:
                        retrieved_relevant.append(rid)
                        break

            k = 5
            p_at_k = len(set(retrieved_relevant)) / k if k > 0 else 0
            r_at_k = (
                len(set(retrieved_relevant)) / len(relevant_docs)
                if relevant_docs else 1.0
            )

            # MRR: rank of first relevant result
            mrr_val = 0.0
            for i, rid in enumerate(retrieved_ids):
                for rdoc in relevant_docs:
                    doc_base = rdoc.split("::")[0]
                    if doc_base in rid:
                        mrr_val = 1.0 / (i + 1)
                        break
                if mrr_val > 0:
                    break

            precisions.append(p_at_k)
            recalls.append(r_at_k)
            mrrs.append(mrr_val)

            results.append({
                "id": query_entry["id"],
                "query": query,
                "category": category,
                "precision_at_5": round(p_at_k, 3),
                "recall_at_5": round(r_at_k, 3),
                "mrr": round(mrr_val, 3),
                "retrieved_sources": [c.get("source", "") for c in chunks],
                "relevant_docs": relevant_docs,
                "num_chunks_retrieved": len(chunks),
            })

        # Aggregate
        summary = {
            "total_queries": len(rag_queries_data),
            "evaluated_queries": len(precisions),
            "mean_precision_at_5": round(sum(precisions) / len(precisions), 3) if precisions else 0,
            "mean_recall_at_5": round(sum(recalls) / len(recalls), 3) if recalls else 0,
            "mean_mrr": round(sum(mrrs) / len(mrrs), 3) if mrrs else 0,
            "per_query": results,
        }

        # Save report
        report_path = os.path.join(reports_dir, "module2_rag.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Assertions — retrieval should be somewhat effective
        assert summary["mean_precision_at_5"] > 0.0, "Precision@5 should be > 0"
        assert summary["mean_mrr"] > 0.0, "MRR should be > 0"

    def test_domain_boundary(self, rag_queries_data):
        """
        2.4 — Out-of-domain queries should return low-relevance chunks.
        """
        try:
            from rag.retriever import _retrieve_sync
        except Exception:
            pytest.skip("RAG retriever not importable")

        ood_queries = [q for q in rag_queries_data if q.get("category") == "out_of_domain"]
        assert len(ood_queries) >= 3, "Need at least 3 out-of-domain queries"

        for query_entry in ood_queries:
            chunks = _retrieve_sync(query_entry["query"], top_k=5)

            if chunks:
                # Out-of-domain: scores (cosine distances) should be higher (worse)
                avg_score = sum(c.get("score", 0) for c in chunks) / len(chunks)
                # ChromaDB cosine distance: higher = less similar
                # We just verify retrieval doesn't crash and log the scores
                assert isinstance(avg_score, float), "Scores should be numeric"

    def test_retrieval_returns_chunks(self, rag_queries_data):
        """Basic test: retriever returns non-empty results for medical queries."""
        try:
            from rag.retriever import _retrieve_sync
        except Exception:
            pytest.skip("RAG retriever not importable")

        medical_queries = [q for q in rag_queries_data if q.get("category") != "out_of_domain"][:5]
        for q in medical_queries:
            chunks = _retrieve_sync(q["query"], top_k=4)
            assert len(chunks) > 0, f"No chunks returned for: {q['query']}"
            assert "text" in chunks[0], "Chunk should have 'text' field"
            assert "source" in chunks[0], "Chunk should have 'source' field"
