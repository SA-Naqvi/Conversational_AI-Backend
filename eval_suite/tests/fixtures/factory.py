"""
Test fixture factory — seeds CRM and vector store with known test data.
Provides setup/teardown for reproducible evaluations.
"""
import os
import sys
import json
import pathlib
import shutil
from datetime import datetime
from typing import List, Dict

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

CRM_FILE = pathlib.Path(BACKEND_DIR) / "data" / "crm" / "users.json"
SESSIONS_DIR = pathlib.Path(BACKEND_DIR) / "data" / "sessions"

# Test session IDs (deterministic for reproducibility)
TEST_SESSION_IDS = [
    "eval-test-session-001",
    "eval-test-session-002",
    "eval-test-session-003",
    "eval-test-session-004",
    "eval-test-session-005",
]

# Known test patient profiles
TEST_PATIENTS = {
    "eval-test-session-001": {
        "name": "Alice Johnson",
        "phone": "555-0101",
        "email": "alice@test.com",
        "medication_allergies": "penicillin",
        "caregiver_name": "Bob Johnson",
        "notes": "Knee replacement patient, day 5",
        "created_at": "2025-01-01T00:00:00",
        "interaction_count": 3,
        "last_updated": "2025-01-03T00:00:00",
        "interaction_history": [
            {"timestamp": "2025-01-01T10:00:00", "note": "Initial intake completed"},
            {"timestamp": "2025-01-02T10:00:00", "note": "Pain reported at 5/10"},
            {"timestamp": "2025-01-03T10:00:00", "note": "Recovery progressing well"},
        ],
    },
    "eval-test-session-002": {
        "name": "Carlos Rivera",
        "phone": "555-0102",
        "created_at": "2025-01-02T00:00:00",
        "interaction_count": 1,
    },
}


def seed_crm() -> List[str]:
    """
    Seed the CRM database with known test patients.
    Returns list of seeded session IDs.
    Preserves any existing non-test records.
    """
    CRM_FILE.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if CRM_FILE.exists():
        try:
            existing = json.loads(CRM_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = {}

    # Add test patients (overwrite if they exist)
    existing.update(TEST_PATIENTS)
    CRM_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

    return list(TEST_PATIENTS.keys())


def seed_vector_store() -> bool:
    """
    Ensure the ChromaDB vector index is built.
    Returns True if index is ready.
    """
    try:
        from rag.indexer import get_client_and_collection
        _, collection = get_client_and_collection()
        count = collection.count()
        if count > 0:
            return True

        # Index is empty — rebuild
        from rag.indexer import build_index
        build_index(force_rebuild=True)
        return True
    except Exception as e:
        print(f"Warning: Could not seed vector store: {e}")
        return False


def teardown(session_ids: List[str] = None) -> int:
    """
    Remove only test-created records from CRM and session files.
    Returns number of records cleaned up.
    """
    ids_to_clean = session_ids or TEST_SESSION_IDS
    cleaned = 0

    # Clean CRM records
    if CRM_FILE.exists():
        try:
            data = json.loads(CRM_FILE.read_text(encoding="utf-8"))
            for sid in ids_to_clean:
                if sid in data:
                    del data[sid]
                    cleaned += 1
            CRM_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    # Clean session files
    for sid in ids_to_clean:
        session_file = SESSIONS_DIR / f"{sid}.json"
        if session_file.exists():
            try:
                session_file.unlink()
                cleaned += 1
            except OSError:
                pass

    return cleaned


def reset_all():
    """Full wipe and rebuild of test fixtures."""
    teardown(TEST_SESSION_IDS)
    seed_crm()
    seed_vector_store()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fixture factory for eval suite")
    parser.add_argument("--seed", action="store_true", help="Seed CRM and vector store")
    parser.add_argument("--teardown", action="store_true", help="Remove test records")
    parser.add_argument("--reset", action="store_true", help="Full reset and rebuild")
    args = parser.parse_args()

    if args.reset:
        reset_all()
        print("Full reset complete.")
    elif args.seed:
        ids = seed_crm()
        print(f"Seeded CRM with {len(ids)} test patients: {ids}")
        ok = seed_vector_store()
        print(f"Vector store ready: {ok}")
    elif args.teardown:
        n = teardown()
        print(f"Cleaned up {n} test records.")
    else:
        parser.print_help()
