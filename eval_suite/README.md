# Nurse GPT-E — Evaluation Suite

Automated evaluation framework for the Medical Post-Op Recovery Companion chatbot.

## Modules

| Module | Focus | LLM Required? |
|--------|-------|---------------|
| **1** | Conversational Correctness — 12 multi-turn dialogues, task completion, policy adherence, state machine accuracy | ✅ Yes |
| **2** | RAG Component — precision@5, recall@5, MRR, domain boundary | ❌ No (retriever only) |
| **3** | CRM Tool — CRUD unit tests, LLM tool-call accuracy | Partial |
| **4** | Additional Tools — OpenFDA/PubMed/Calendar functional tests, LLM invocation accuracy, confusion matrix | Partial |
| **5** | Latency & Throughput — TTFT, inter-token, end-to-end, concurrency ramp | ✅ Yes |

## Quick Start

```bash
# 1. Install eval dependencies
cd eval_suite
pip install -r requirements.txt

# 2. Start the backend server (in another terminal)
cd ../
uvicorn main:app --host 0.0.0.0 --port 8000

# 3. Run all evaluations
python run_evals.py

# 4. Run only unit tests (no LLM needed)
python run_evals.py --skip-llm

# 5. Run a specific module
python run_evals.py --module 2

# 6. Full performance benchmarks (30 trials)
python run_evals.py --full-perf
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--skip-llm` | Skip all tests requiring the LLM server |
| `--full-perf` | Run 30 trials per performance scenario (default: 3) |
| `--reset` | Force rebuild of all test fixtures |
| `--module N` | Run only module N (1–5) |
| `--report-only` | Skip tests, regenerate report from existing results |

## Output

Reports are generated in `eval_suite/reports/`:

| File | Description |
|------|-------------|
| `eval_report.json` | Raw results (all modules) |
| `eval_report.md` | Markdown summary with tables |
| `eval_report.html` | Standalone HTML report |
| `latency.png` | Box plot of latency by scenario |
| `concurrency.png` | Latency vs concurrent users |
| `confusion_matrix.png` | Tool selection confusion matrix |

## Load Testing with Locust

```bash
cd eval_suite
locust -f locustfile.py --host=ws://localhost:8000 --users 10 --spawn-rate 2
```

Then open http://localhost:8089 for the Locust web UI.

## Ground Truth Data

Test data files in `tests/data/`:

- **conversations.json** — 12 multi-turn dialogues with expected states and escalation flags
- **rag_queries.json** — 34 queries (29 medical + 5 out-of-domain) with relevant document annotations
- **tool_invocations.json** — 28 utterances across 5 categories with expected tool mappings
