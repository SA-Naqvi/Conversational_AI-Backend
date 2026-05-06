#!/usr/bin/env python3
"""
Evaluation Suite Runner & Report Generator.

Entry point for running all evaluation modules and generating
JSON + Markdown + HTML reports.

Usage:
  python run_evals.py                      # Run all modules
  python run_evals.py --skip-llm           # Skip LLM-dependent tests
  python run_evals.py --module 2           # Run only Module 2 (RAG)
  python run_evals.py --full-perf          # 30 trials per perf scenario
  python run_evals.py --reset              # Rebuild fixtures
"""
import os
import sys
import json
import subprocess
import argparse
import datetime
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

EVAL_DIR = Path(__file__).parent
BACKEND_DIR = EVAL_DIR.parent
REPORTS_DIR = EVAL_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def check_server(host="localhost", port=8000) -> bool:
    """Check if the backend server is reachable."""
    try:
        import httpx
        resp = httpx.get(f"http://{host}:{port}/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def run_fixtures(reset: bool = False):
    """Seed test fixtures."""
    factory_path = EVAL_DIR / "tests" / "fixtures" / "factory.py"
    cmd = [sys.executable, str(factory_path)]
    cmd.append("--reset" if reset else "--seed")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(BACKEND_DIR), env=env)
    print(result.stdout)
    if result.returncode != 0:
        print(f"⚠ Fixture warning: {result.stderr}")


def run_pytest(module: int = None, skip_llm: bool = False, full_perf: bool = False) -> int:
    """Run pytest for the specified module(s)."""
    # Module filtering
    module_map = {
        1: "test_conversations.py",
        2: "test_rag.py",
        3: "test_crm.py",
        4: "test_tools.py",
        5: "test_performance.py",
    }

    if module and module in module_map:
        test_path = str(EVAL_DIR / "tests" / module_map[module])
    else:
        test_path = str(EVAL_DIR / "tests")

    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--tb=short",
        f"--rootdir={BACKEND_DIR}",
        "-p", "no:cacheprovider",
    ]

    if skip_llm:
        cmd.append("--skip-llm")
    if full_perf:
        cmd.append("--full-perf")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND_DIR) + os.pathsep + env.get("PYTHONPATH", "")
    
    result = subprocess.run(cmd, capture_output=False, text=True, cwd=str(BACKEND_DIR), env=env)
    return result.returncode


def collect_results() -> dict:
    """Collect all module result JSON files."""
    results = {
        "timestamp": datetime.datetime.now().isoformat(),
        "modules": {},
    }

    module_files = {
        "module1_conversations": "Module 1: Conversational Correctness",
        "module2_rag": "Module 2: RAG Component",
        "module3_crm": "Module 3: CRM Tool",
        "module4_tools": "Module 4: Additional Tools",
        "module5_performance": "Module 5: Latency (Single-User)",
        "module5_concurrency": "Module 5: Concurrency Ramp",
        "hardware_context": "Hardware Context",
    }

    for filename, title in module_files.items():
        filepath = REPORTS_DIR / f"{filename}.json"
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    results["modules"][filename] = {
                        "title": title,
                        "data": json.load(f),
                    }
            except Exception as e:
                results["modules"][filename] = {"title": title, "error": str(e)}

    return results


def generate_markdown_report(results: dict) -> str:
    """Generate a Markdown summary report."""
    lines = [
        "# 🏥 Nurse GPT-E — Evaluation Suite Report",
        "",
        f"**Generated:** {results['timestamp']}",
        "",
        "---",
        "",
    ]

    # Hardware Context
    hw = results["modules"].get("hardware_context", {}).get("data", {})
    if hw:
        lines.extend([
            "## 🖥️ Hardware Context",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Platform | {hw.get('platform', 'N/A')} |",
            f"| Python | {hw.get('python_version', 'N/A')} |",
            f"| CPU | {hw.get('processor', 'N/A')} |",
            f"| Cores | {hw.get('cpu_count', 'N/A')} |",
            f"| RAM | {hw.get('ram_gb', 'N/A')} GB |",
            "",
        ])

    # Module 1: Conversations
    m1 = results["modules"].get("module1_conversations", {}).get("data", {})
    if m1:
        lines.extend([
            "## 💬 Module 1: Conversational Correctness",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Conversations | {m1.get('total_conversations', 'N/A')} |",
            f"| Task Completion Rate | {m1.get('task_completion_rate', 0):.1%} |",
            f"| Escalation Accuracy | {m1.get('escalation_accuracy', 0):.1%} |",
            f"| Mean State Accuracy | {m1.get('mean_state_accuracy', 'N/A')} |",
            "",
        ])

        # Per-conversation table
        per_conv = m1.get("per_conversation", [])
        if per_conv:
            lines.extend([
                "### Per-Conversation Results",
                "",
                "| ID | Task Complete | Escalation OK | State Accuracy |",
                "|----|---------------|---------------|----------------|",
            ])
            for c in per_conv:
                tc = "✅" if c.get("task_completion") else "❌"
                ec = "✅" if c.get("escalation_correct") else "❌"
                sa = f"{c.get('state_machine_accuracy', 'N/A')}"
                lines.append(f"| {c.get('conversation_id', 'N/A')} | {tc} | {ec} | {sa} |")
            lines.append("")

    # Module 2: RAG
    m2 = results["modules"].get("module2_rag", {}).get("data", {})
    if m2:
        lines.extend([
            "## 📚 Module 2: RAG Component",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Queries | {m2.get('total_queries', 'N/A')} |",
            f"| Evaluated Queries | {m2.get('evaluated_queries', 'N/A')} |",
            f"| Mean Precision@5 | {m2.get('mean_precision_at_5', 'N/A')} |",
            f"| Mean Recall@5 | {m2.get('mean_recall_at_5', 'N/A')} |",
            f"| Mean MRR | {m2.get('mean_mrr', 'N/A')} |",
            "",
        ])

    # Module 3: CRM
    m3 = results["modules"].get("module3_crm", {}).get("data", {})
    if m3:
        lines.extend([
            "## 🗂️ Module 3: CRM Tool",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {m3.get('total_tests', 'N/A')} |",
            f"| True Positives | {m3.get('true_positives', 'N/A')} |",
            f"| False Positives | {m3.get('false_positives', 'N/A')} |",
            f"| False Negatives | {m3.get('false_negatives', 'N/A')} |",
            f"| Precision | {m3.get('precision', 'N/A')} |",
            f"| Recall | {m3.get('recall', 'N/A')} |",
            "",
        ])

    # Module 4: Tools
    m4 = results["modules"].get("module4_tools", {}).get("data", {})
    if m4:
        lines.extend([
            "## 🔧 Module 4: Tool Invocation",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Tests | {m4.get('total_tests', 'N/A')} |",
            f"| Correct | {m4.get('correct', 'N/A')} |",
            f"| Accuracy | {m4.get('accuracy', 'N/A')} |",
            "",
        ])

        # Confusion matrix
        cm = m4.get("confusion_matrix", {})
        if cm:
            tool_names = m4.get("tool_names", sorted(set(
                list(cm.keys()) + [k2 for v in cm.values() for k2 in v.keys()]
            )))
            lines.extend([
                "### Confusion Matrix",
                "",
                "| Expected \\ Actual | " + " | ".join(tool_names) + " |",
                "|" + "|".join(["---"] * (len(tool_names) + 1)) + "|",
            ])
            for expected in tool_names:
                row = [expected]
                for actual in tool_names:
                    row.append(str(cm.get(expected, {}).get(actual, 0)))
                lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    # Module 5: Performance
    m5 = results["modules"].get("module5_performance", {}).get("data", {})
    if m5:
        lines.extend([
            "## ⚡ Module 5: Latency & Throughput",
            "",
        ])
        scenarios = m5.get("scenarios", {})
        if scenarios:
            lines.extend([
                "### Single-User Latency",
                "",
                "| Scenario | Trials | TTFT (med) | Total (med) | Inter-token (med) |",
                "|----------|--------|------------|-------------|-------------------|",
            ])
            for name, data in scenarios.items():
                trials = data.get("successful_trials", 0)
                ttft_med = data.get("ttft_ms", {}).get("median", "N/A")
                total_med = data.get("total_ms", {}).get("median", "N/A")
                it_med = data.get("inter_token_ms", {}).get("median", "N/A")
                lines.append(
                    f"| {name} | {trials} | {ttft_med} ms | {total_med} ms | {it_med} ms |"
                )
            lines.append("")

    # Concurrency
    m5c = results["modules"].get("module5_concurrency", {}).get("data", {})
    if m5c:
        ramp = m5c.get("ramp_results", [])
        if ramp:
            lines.extend([
                "### Concurrency Ramp",
                "",
                "| Users | Wall Time | Median TTFT | Median Total | Errors |",
                "|-------|-----------|-------------|--------------|--------|",
            ])
            for r in ramp:
                lines.append(
                    f"| {r['concurrent_users']} | {r.get('wall_time_ms', 'N/A')} ms | "
                    f"{r.get('median_ttft_ms', 'N/A')} ms | "
                    f"{r.get('median_total_ms', 'N/A')} ms | {r.get('errors', 0)} |"
                )
            lines.append("")

    lines.extend([
        "---",
        "",
        "*Report generated by the Nurse GPT-E Evaluation Suite*",
    ])

    return "\n".join(lines)


def generate_html_report(markdown_content: str) -> str:
    """Generate a standalone HTML report from markdown content."""
    # Simple HTML wrapper with styling
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Nurse GPT-E — Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 2rem;
            background: #0f1117;
            color: #e0e0e0;
            line-height: 1.6;
        }}
        h1 {{ color: #60a5fa; border-bottom: 2px solid #1e3a5f; padding-bottom: 0.5rem; }}
        h2 {{ color: #93c5fd; margin-top: 2rem; }}
        h3 {{ color: #bfdbfe; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
            font-size: 0.9rem;
        }}
        th, td {{
            border: 1px solid #2d3748;
            padding: 0.5rem 0.75rem;
            text-align: left;
        }}
        th {{ background: #1a2332; color: #93c5fd; }}
        tr:nth-child(even) {{ background: #141922; }}
        tr:hover {{ background: #1e293b; }}
        hr {{ border: none; border-top: 1px solid #2d3748; margin: 2rem 0; }}
        code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 0.85rem; }}
        .pass {{ color: #4ade80; }}
        .fail {{ color: #f87171; }}
    </style>
</head>
<body>
    <div id="content">
"""
    # Convert markdown to HTML (simple conversion)
    content = markdown_content
    # Headers
    for i in range(3, 0, -1):
        prefix = "#" * i
        content = content.replace(f"\n{prefix} ", f"\n<h{i}>")
        lines_out = []
        for line in content.split("\n"):
            if line.startswith(f"<h{i}>"):
                line = line + f"</h{i}>"
            lines_out.append(line)
        content = "\n".join(lines_out)

    # Tables
    in_table = False
    table_lines = []
    result_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            if not in_table:
                in_table = True
                table_lines = []
            if all(c in "-| " for c in stripped):
                continue  # Skip separator row
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if not table_lines:
                result_lines.append("<table><thead><tr>" +
                                   "".join(f"<th>{c}</th>" for c in cells) +
                                   "</tr></thead><tbody>")
            else:
                # Replace emoji checkmarks
                cells = [c.replace("✅", '<span class="pass">✅</span>').replace("❌", '<span class="fail">❌</span>') for c in cells]
                result_lines.append("<tr>" +
                                   "".join(f"<td>{c}</td>" for c in cells) +
                                   "</tr>")
            table_lines.append(line)
        else:
            if in_table:
                result_lines.append("</tbody></table>")
                in_table = False
                table_lines = []
            if stripped == "---":
                result_lines.append("<hr>")
            elif stripped.startswith("**") and stripped.endswith("**"):
                result_lines.append(f"<p><strong>{stripped[2:-2]}</strong></p>")
            elif stripped:
                result_lines.append(f"<p>{stripped}</p>")

    if in_table:
        result_lines.append("</tbody></table>")

    html += "\n".join(result_lines)
    html += """
    </div>
</body>
</html>"""
    return html


def generate_charts(results: dict):
    """Generate performance charts using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("⚠ matplotlib not available — skipping chart generation")
        return

    # Latency box plot
    m5 = results["modules"].get("module5_performance", {}).get("data", {})
    if m5 and m5.get("scenarios"):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.patch.set_facecolor("#0f1117")

        scenarios = m5["scenarios"]
        names = list(scenarios.keys())

        # TTFT box plot
        ax = axes[0]
        ax.set_facecolor("#141922")
        ttft_data = []
        for name in names:
            raw = scenarios[name].get("raw_trials", [])
            ttft_data.append([t.get("ttft_ms", 0) for t in raw if t.get("ttft_ms") is not None])

        if any(ttft_data):
            bp = ax.boxplot(ttft_data, labels=names, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#1e3a5f")
                patch.set_edgecolor("#60a5fa")
            ax.set_title("TTFT by Scenario", color="#e0e0e0", fontsize=12)
            ax.set_ylabel("TTFT (ms)", color="#e0e0e0")
            ax.tick_params(colors="#e0e0e0")

        # Total latency box plot
        ax = axes[1]
        ax.set_facecolor("#141922")
        total_data = []
        for name in names:
            raw = scenarios[name].get("raw_trials", [])
            total_data.append([t.get("total_ms", 0) for t in raw if t.get("total_ms") is not None])

        if any(total_data):
            bp = ax.boxplot(total_data, labels=names, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("#1e3a5f")
                patch.set_edgecolor("#60a5fa")
            ax.set_title("Total Latency by Scenario", color="#e0e0e0", fontsize=12)
            ax.set_ylabel("Total (ms)", color="#e0e0e0")
            ax.tick_params(colors="#e0e0e0")

        plt.tight_layout()
        plt.savefig(str(REPORTS_DIR / "latency.png"), dpi=150, facecolor="#0f1117")
        plt.close()
        print("📊 Generated latency.png")

    # Concurrency ramp chart
    m5c = results["modules"].get("module5_concurrency", {}).get("data", {})
    if m5c and m5c.get("ramp_results"):
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#141922")

        ramp = m5c["ramp_results"]
        users = [r["concurrent_users"] for r in ramp]
        ttfts = [r.get("median_ttft_ms") or 0 for r in ramp]
        totals = [r.get("median_total_ms") or 0 for r in ramp]

        ax.plot(users, ttfts, "o-", color="#60a5fa", label="Median TTFT")
        ax.plot(users, totals, "s-", color="#f87171", label="Median Total")
        ax.set_xlabel("Concurrent Users", color="#e0e0e0")
        ax.set_ylabel("Latency (ms)", color="#e0e0e0")
        ax.set_title("Concurrency Ramp — Latency vs Users", color="#e0e0e0", fontsize=12)
        ax.legend(facecolor="#1a2332", edgecolor="#2d3748", labelcolor="#e0e0e0")
        ax.tick_params(colors="#e0e0e0")
        ax.grid(True, alpha=0.2, color="#2d3748")

        plt.tight_layout()
        plt.savefig(str(REPORTS_DIR / "concurrency.png"), dpi=150, facecolor="#0f1117")
        plt.close()
        print("📊 Generated concurrency.png")

    # Confusion matrix heatmap
    m4 = results["modules"].get("module4_tools", {}).get("data", {})
    if m4 and m4.get("confusion_matrix"):
        cm = m4["confusion_matrix"]
        tool_names = m4.get("tool_names", sorted(set(
            list(cm.keys()) + [k2 for v in cm.values() for k2 in v.keys()]
        )))

        matrix = []
        for expected in tool_names:
            row = []
            for actual in tool_names:
                row.append(cm.get(expected, {}).get(actual, 0))
            matrix.append(row)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#141922")

        im = ax.imshow(matrix, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(tool_names)))
        ax.set_yticks(range(len(tool_names)))
        ax.set_xticklabels(tool_names, rotation=45, ha="right", fontsize=8, color="#e0e0e0")
        ax.set_yticklabels(tool_names, fontsize=8, color="#e0e0e0")
        ax.set_xlabel("Actual Tool", color="#e0e0e0")
        ax.set_ylabel("Expected Tool", color="#e0e0e0")
        ax.set_title("Tool Selection Confusion Matrix", color="#e0e0e0", fontsize=12)

        # Add text annotations
        for i in range(len(tool_names)):
            for j in range(len(tool_names)):
                val = matrix[i][j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center",
                            color="white" if val > max(max(row) for row in matrix) / 2 else "#93c5fd",
                            fontsize=10, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Count")
        plt.tight_layout()
        plt.savefig(str(REPORTS_DIR / "confusion_matrix.png"), dpi=150, facecolor="#0f1117")
        plt.close()
        print("📊 Generated confusion_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="Nurse GPT-E Evaluation Suite Runner")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip tests requiring the LLM server")
    parser.add_argument("--full-perf", action="store_true",
                        help="Run full 30-trial performance benchmarks")
    parser.add_argument("--reset", action="store_true",
                        help="Force rebuild of all fixtures")
    parser.add_argument("--module", type=int, choices=[1, 2, 3, 4, 5],
                        help="Run only a specific module")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip tests, only generate report from existing results")
    args = parser.parse_args()

    print("=" * 60)
    print("🏥 Nurse GPT-E — Evaluation Suite")
    print("=" * 60)
    print()

    if not args.report_only:
        # Check server
        if not args.skip_llm:
            print("🔍 Checking backend server...")
            if not check_server():
                print("❌ Backend server not running at http://localhost:8000")
                print("   Start it with: cd Conversational_AI-Backend && uvicorn main:app --port 8000")
                print("   Or use --skip-llm to skip LLM-dependent tests")
                sys.exit(1)
            print("✅ Server is running")
        print()

        # Seed fixtures
        print("🌱 Seeding test fixtures...")
        run_fixtures(reset=args.reset)
        print()

        # Run tests
        print("🧪 Running evaluation tests...")
        print("-" * 60)
        exit_code = run_pytest(
            module=args.module,
            skip_llm=args.skip_llm,
            full_perf=args.full_perf,
        )
        print("-" * 60)
        print(f"\n{'✅' if exit_code == 0 else '⚠'} Pytest finished with exit code {exit_code}")
        print()

    # Collect results and generate reports
    print("📄 Generating reports...")
    results = collect_results()

    # Save JSON
    json_path = REPORTS_DIR / "eval_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   📁 {json_path}")

    # Generate Markdown
    md_content = generate_markdown_report(results)
    md_path = REPORTS_DIR / "eval_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"   📁 {md_path}")

    # Generate HTML
    html_content = generate_html_report(md_content)
    html_path = REPORTS_DIR / "eval_report.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"   📁 {html_path}")

    # Generate charts
    generate_charts(results)

    print()
    print("=" * 60)
    print("✅ Evaluation complete!")
    print(f"   Reports: {REPORTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
