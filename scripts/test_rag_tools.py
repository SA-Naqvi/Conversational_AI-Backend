"""
Comprehensive test script for:
  1. RAG retrieval (ChromaDB + all-MiniLM-L6-v2)
  2. OpenFDA API (drug labels, adverse events, recalls)
  3. PubMed API (research paper search)
  4. Google Calendar API (event creation/listing — requires credentials)
  5. CRM tool (user profile storage)
  6. Multi-turn conversation simulation

Run from backend/:
    python scripts/test_rag_tools.py

Run individual tests:
    python scripts/test_rag_tools.py --rag
    python scripts/test_rag_tools.py --openfda
    python scripts/test_rag_tools.py --pubmed
    python scripts/test_rag_tools.py --calendar
    python scripts/test_rag_tools.py --crm
    python scripts/test_rag_tools.py --chat
"""
import asyncio
import sys
import pathlib
import argparse
import json
import time
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

# Colors for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def header(title: str):
    print(f"\n{BOLD}{CYAN}{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}{RESET}\n")


def success(msg: str):
    print(f"{GREEN}[OK] {msg}{RESET}")


def warn(msg: str):
    print(f"{YELLOW}[WARN] {msg}{RESET}")


def fail(msg: str):
    print(f"{RED}[FAIL] {msg}{RESET}")


def info(msg: str):
    print(f"  {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAG RETRIEVAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

async def test_rag():
    header("RAG RETRIEVAL TESTS")

    from rag.retriever import retrieve, format_context

    test_queries = [
        ("what should I eat after surgery", "diet"),
        ("how to manage pain after knee replacement", "pain"),
        ("signs of wound infection", "infection"),
        ("when can I drive after hip surgery", "activity"),
        ("what are the symptoms of deep vein thrombosis", "DVT"),
    ]

    total_time = 0
    for query, expected_topic in test_queries:
        start = time.time()
        chunks = await retrieve(query, top_k=3)
        elapsed = (time.time() - start) * 1000
        total_time += elapsed

        if chunks:
            success(f"Query: '{query[:40]}...'")
            info(f"  Retrieved {len(chunks)} chunks in {elapsed:.0f}ms")
            for i, c in enumerate(chunks, 1):
                score = c["score"]
                src = c["source"][:30]
                preview = c["text"][:60].replace("\n", " ")
                info(f"    [{i}] score={score:.3f} | {src} | {preview}...")
        else:
            fail(f"Query: '{query}' returned no results")

    avg_time = total_time / len(test_queries)
    print()
    success(f"Average retrieval time: {avg_time:.0f}ms")

    # Test context formatting
    chunks = await retrieve("post surgery diet and nutrition", top_k=4)
    context = format_context(chunks)
    if context and "[RETRIEVED MEDICAL KNOWLEDGE]" in context:
        success(f"Context formatting OK ({len(context)} chars)")
    else:
        fail("Context formatting failed")


# ══════════════════════════════════════════════════════════════════════════════
# 2. OPENFDA API TESTS
# ══════════════════════════════════════════════════════════════════════════════

async def test_openfda():
    header("OPENFDA API TESTS")

    from tools.openfda_tool import DrugLabelTool, DrugAdverseEventsTool, DrugRecallTool

    # Test drug label lookup
    print(f"{BOLD}Drug Label Lookup:{RESET}")
    label_tool = DrugLabelTool()

    test_drugs = ["ibuprofen", "oxycodone", "amoxicillin"]
    for drug in test_drugs:
        start = time.time()
        result = await label_tool.execute(drug_name=drug)
        elapsed = (time.time() - start) * 1000

        if result.get("found"):
            success(f"{drug}: found ({elapsed:.0f}ms)")
            info(f"  Generic: {result.get('generic_names', ['?'])[0]}")
            info(f"  Brand: {result.get('brand_names', ['?'])[:2]}")
            if result.get("warnings"):
                info(f"  Warnings: {result['warnings'][:100]}...")
        else:
            warn(f"{drug}: {result.get('message', 'not found')}")

    # Test adverse events
    print(f"\n{BOLD}Adverse Events Search:{RESET}")
    adverse_tool = DrugAdverseEventsTool()

    result = await adverse_tool.execute(drug_name="aspirin", top_n=5)
    if result.get("found"):
        success(f"Aspirin adverse events: {result['total_reports_sampled']} reports sampled")
        for reaction in result.get("top_adverse_reactions", [])[:5]:
            info(f"  - {reaction['reaction']}: {reaction['reports']} reports")
    else:
        warn(f"Aspirin adverse events: {result.get('message', 'not found')}")

    # Test recall check
    print(f"\n{BOLD}Drug Recall Check:{RESET}")
    recall_tool = DrugRecallTool()
    result = await recall_tool.execute(drug_name="metformin")
    if result.get("found"):
        success(f"Metformin recalls found: {result['recall_count']}")
        for r in result.get("recalls", [])[:2]:
            info(f"  - {r['reason'][:80]}...")
    else:
        info(f"No metformin recalls found (this is good news)")


# ══════════════════════════════════════════════════════════════════════════════
# 3. PUBMED API TESTS
# ══════════════════════════════════════════════════════════════════════════════

async def test_pubmed():
    header("PUBMED API TESTS")

    from tools.pubmed_tool import PubMedSearchTool, PubMedGetAbstractTool

    search_tool = PubMedSearchTool()

    test_queries = [
        "post-operative pain management after total knee replacement",
        "deep vein thrombosis prevention after surgery",
        "wound healing stages",
    ]

    for query in test_queries:
        print(f"{BOLD}Query:{RESET} {query[:50]}...")
        start = time.time()
        result = await search_tool.execute(query=query, max_results=3, years_back=10)
        elapsed = (time.time() - start) * 1000

        if result.get("found"):
            success(f"Found {result['result_count']} articles ({elapsed:.0f}ms)")
            for article in result.get("articles", []):
                info(f"  [{article['year']}] {article['title'][:60]}...")
                info(f"       Authors: {article['authors']}")
                info(f"       URL: {article['url']}")
        else:
            warn(f"No results: {result.get('message', 'unknown error')}")
        print()

    # Test specific abstract retrieval
    print(f"{BOLD}Abstract Retrieval Test:{RESET}")
    abstract_tool = PubMedGetAbstractTool()
    # Use a known PMID (a review article on post-surgical pain)
    result = await abstract_tool.execute(pmid="35912345")
    if result.get("found"):
        success(f"Retrieved abstract for PMID {result['pmid']}")
        info(f"  Title: {result['title'][:70]}...")
    else:
        info(f"PMID not found (expected for test PMID)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. GOOGLE CALENDAR API TESTS
# ══════════════════════════════════════════════════════════════════════════════

async def test_calendar():
    header("GOOGLE CALENDAR API TESTS")

    from tools.google_calendar_tool import (
        CreateCalendarEventTool,
        ListCalendarEventsTool,
        DeleteCalendarEventTool,
        _is_configured,
    )

    if not _is_configured():
        warn("Google Calendar not configured (credentials.json missing)")
        info("To enable:")
        info("  1. Create OAuth credentials at console.cloud.google.com")
        info("  2. Enable 'Google Calendar API'")
        info("  3. Download credentials.json to: backend/data/calendar/credentials.json")
        info("  4. Set GOOGLE_CALENDAR_ID=primary in .env")
        info("  5. Run this test again — a browser will open for authorization")
        return

    # List existing events
    print(f"{BOLD}Listing Upcoming Events:{RESET}")
    list_tool = ListCalendarEventsTool()
    result = await list_tool.execute(max_results=5, days_ahead=14)
    if result.get("found"):
        success(f"Found {result['event_count']} upcoming events")
        for ev in result.get("events", []):
            info(f"  - {ev['summary']} @ {ev['start']}")
    else:
        info("No upcoming events (or calendar empty)")

    # Create a test event
    print(f"\n{BOLD}Creating Test Event:{RESET}")
    create_tool = CreateCalendarEventTool()
    result = await create_tool.execute(
        summary="[TEST] Post-Op Follow-Up — Dr. Smith",
        date="next Monday",
        time="10:00 AM",
        duration_minutes=30,
        description="Automated test event from Nurse GPT-E. Safe to delete.",
        send_reminder=False,
    )

    if result.get("success"):
        event_id = result["event_id"]
        success(f"Event created: {result['summary']}")
        info(f"  ID: {event_id}")
        info(f"  Time: {result['start']}")
        info(f"  Link: {result.get('calendar_link', 'N/A')}")

        # Delete the test event
        print(f"\n{BOLD}Deleting Test Event:{RESET}")
        delete_tool = DeleteCalendarEventTool()
        del_result = await delete_tool.execute(event_id=event_id)
        if del_result.get("success"):
            success(f"Test event deleted successfully")
        else:
            warn(f"Failed to delete: {del_result.get('error')}")
    else:
        warn(f"Failed to create event: {result.get('error')}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. CRM TOOL TESTS
# ══════════════════════════════════════════════════════════════════════════════

async def test_crm():
    header("CRM TOOL TESTS")

    from tools.crm_tool import GetUserInfoTool, UpdateUserInfoTool, RecordInteractionTool

    test_session = f"test-session-{datetime.now().strftime('%Y%m%d%H%M%S')}"

    # Update user info
    print(f"{BOLD}Storing User Profile:{RESET}")
    update_tool = UpdateUserInfoTool()

    fields = [
        ("name", "Jane Doe"),
        ("phone", "555-123-4567"),
        ("surgery_type", "Total Knee Replacement"),
        ("medication_allergies", "Penicillin"),
        ("preferred_contact", "Phone"),
    ]

    for field, value in fields:
        result = await update_tool.execute(session_id=test_session, field=field, value=value)
        if result.get("success"):
            success(f"Stored {field}={value}")
        else:
            fail(f"Failed to store {field}")

    # Record interaction
    print(f"\n{BOLD}Recording Interaction:{RESET}")
    record_tool = RecordInteractionTool()
    result = await record_tool.execute(
        session_id=test_session,
        note="Patient reported pain level 4/10, improving from yesterday's 6/10",
    )
    if result.get("success"):
        success(f"Interaction recorded: {result['note_recorded'][:50]}...")
    else:
        fail("Failed to record interaction")

    # Retrieve user info
    print(f"\n{BOLD}Retrieving User Profile:{RESET}")
    get_tool = GetUserInfoTool()
    result = await get_tool.execute(session_id=test_session)
    if result.get("found"):
        success(f"Profile retrieved for session {test_session[:20]}...")
        user = result["user"]
        for key, val in user.items():
            if key not in ("created_at", "last_updated", "last_seen"):
                info(f"  {key}: {val}")
    else:
        fail("Failed to retrieve profile")


# ══════════════════════════════════════════════════════════════════════════════
# 6. MULTI-TURN CONVERSATION SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

async def test_multi_turn_chat():
    header("MULTI-TURN CONVERSATION SIMULATION")

    from conversation_manager.manager import handle_message
    import uuid

    session_id = str(uuid.uuid4())

    conversation = [
        # Turn 1: Greeting and basic info
        "Hi, I had knee replacement surgery 5 days ago and I'm feeling some pain.",
        # Turn 2: Ask about medication
        "Can I take ibuprofen for the pain? What's the right dosage?",
        # Turn 3: Ask about side effects (triggers OpenFDA)
        "What are the common side effects of ibuprofen?",
        # Turn 4: Ask about recovery timeline (triggers RAG)
        "How long does swelling typically last after knee replacement?",
        # Turn 5: Ask research-based question (triggers PubMed)
        "Is there any research on how physical therapy helps knee replacement recovery?",
        # Turn 6: Schedule appointment (triggers Calendar — may fail gracefully if not configured)
        "Can you schedule a follow-up appointment for next Monday at 10am?",
        # Turn 7: Personal info storage (triggers CRM)
        "My name is John Smith and my phone number is 555-987-6543.",
    ]

    print(f"Session ID: {session_id[:8]}...\n")

    for i, user_msg in enumerate(conversation, 1):
        print(f"{BOLD}[Turn {i}] User:{RESET}")
        print(f"  {user_msg}\n")

        print(f"{BOLD}[Turn {i}] Assistant:{RESET}")
        full_response = ""
        start = time.time()

        try:
            async for chunk in handle_message(session_id, user_msg):
                full_response += chunk
                # Print streaming chunks (simulate real-time)
                print(chunk, end="", flush=True)

            elapsed = (time.time() - start) * 1000
            print(f"\n{CYAN}  [Response time: {elapsed:.0f}ms, {len(full_response)} chars]{RESET}")

        except Exception as e:
            print(f"\n{RED}  [Error: {e}]{RESET}")

        print("\n" + "-" * 60 + "\n")

        # Small delay between turns
        await asyncio.sleep(0.5)

    success(f"Multi-turn conversation completed ({len(conversation)} turns)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="Test RAG and Tools")
    parser.add_argument("--rag", action="store_true", help="Run RAG tests only")
    parser.add_argument("--openfda", action="store_true", help="Run OpenFDA tests only")
    parser.add_argument("--pubmed", action="store_true", help="Run PubMed tests only")
    parser.add_argument("--calendar", action="store_true", help="Run Google Calendar tests only")
    parser.add_argument("--crm", action="store_true", help="Run CRM tests only")
    parser.add_argument("--chat", action="store_true", help="Run multi-turn chat simulation only")
    args = parser.parse_args()

    # If no specific flag, run all tests except chat (which requires LLM)
    run_all = not any([args.rag, args.openfda, args.pubmed, args.calendar, args.crm, args.chat])

    print(f"\n{BOLD}{CYAN}+============================================================+")
    print(f"|     MEDICAL CHATBOT - RAG + TOOLS TEST SUITE             |")
    print(f"|     {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                              |")
    print(f"+============================================================+{RESET}")

    if args.rag or run_all:
        await test_rag()

    if args.openfda or run_all:
        await test_openfda()

    if args.pubmed or run_all:
        await test_pubmed()

    if args.calendar or run_all:
        await test_calendar()

    if args.crm or run_all:
        await test_crm()

    if args.chat:
        await test_multi_turn_chat()

    print(f"\n{BOLD}{GREEN}All tests completed!{RESET}\n")


if __name__ == "__main__":
    asyncio.run(main())
