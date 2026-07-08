import asyncio
import json
import os
import sys
from pathlib import Path

# Prevent charmap encoding errors on Windows console
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root and all component directories to sys.path
project_root = r"C:\Users\omsai\PycharmProjects\PaperForge"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeCode")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeWorkspace")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeKnowledge")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeResearch")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\PaperUnderstanding")

from core import forge_project
from adapters.workspace_adapter import WorkspaceAdapter
from adapters.knowledge_adapter import KnowledgeAdapter
from adapters.research_adapter import ResearchAdapter
from adapters.understanding_adapter import UnderstandingAdapter

# Import REAL functions from components
from workspace.project import initialize_project
from workspace.files import write_project_file
from workspace.verifer import verify_project

from retrieval.search import search_documents, find_exact_symbol

from research.discovery import find_research_papers
from research.inspector import inspect_research_page

from client import analyze_paper


workspace = WorkspaceAdapter(
    initialize_fn=initialize_project,
    write_fn=write_project_file,
    verify_fn=verify_project,
)

knowledge = KnowledgeAdapter(
    search_fn=search_documents,
    exact_symbol_fn=find_exact_symbol,
)

research = ResearchAdapter(
    find_papers_fn=find_research_papers,
    inspect_page_fn=inspect_research_page,
)

understanding = UnderstandingAdapter(
    analyze_paper_fn=analyze_paper,
)


async def run_paper_to_code(pdf_filename: str, max_review_rounds: int = 3, dataset_name: str | None = None):
    paper_dir = Path(project_root) / "PaperUnderstanding" / "papers"
    pdf_path = paper_dir / pdf_filename

    if not pdf_path.exists():
        print(f"[ERROR] Paper not found at: {pdf_path}")
        print(f"Please put your PDF paper into: {paper_dir}")
        return

    print("\n==========================================")
    print("      PAPERFORGE: PAPER TO CODE ENGINE    ")
    print("==========================================")
    print(f"1. Reading paper: {pdf_filename} from PaperUnderstanding/papers/...")

    result = await forge_project(
        pdf_path=str(pdf_path),
        workspace=workspace,
        knowledge=knowledge,
        research=research,
        understanding=understanding,
        max_review_rounds=max_review_rounds,
        dataset_name=dataset_name,
    )

    print("\n2. Extracted Details (PaperUnderstanding):")
    under_res = result.get("understanding_result") or {}
    print(f"   - Title: {under_res.get('paper_title', 'Unknown')}")
    print(f"   - Task: {under_res.get('task', 'Unknown')}")
    print(f"   - Extracted Components: {len(under_res.get('model_components', []))}")
    for c in under_res.get("model_components", []):
        print(f"     * {c.get('name')} ({c.get('component_type')}): {len(c.get('facts', []))} facts")

    routing = result.get("routing_evidence", {})
    print("\n3. Grounded Evidence Bundle (ForgeCore Router):")
    print(f"   - Routed to Knowledge: {routing.get('routed_to_knowledge', False)} ({len(routing.get('knowledge_evidence', []))} API docs)")
    print(f"   - Routed to Research: {routing.get('routed_to_research', False)} ({len(routing.get('research_evidence', []))} papers)")

    res_report = result.get("resolution_report") or {}
    if res_report.get("resolved_count", 0) > 0:
        tiers = res_report.get("tiers", {})
        print(f"\n   [NullResolver] Resolved {res_report['resolved_count']} gaps: "
              f"{tiers.get('registry', 0)} REGISTRY_CANONICAL, "
              f"{tiers.get('literature', 0)} LITERATURE_GROUNDED, "
              f"{tiers.get('heuristic', 0)} DOMAIN_HEURISTIC")
        print(f"   [NullResolver] Detected dataset: {res_report.get('detected_dataset', 'unknown')}")

    print("\n4. Implementing Paper in ForgeWorkspace:")
    code_res = result.get("code_result", {})
    print("   [ForgeCode] Semantic review ->", code_res.get("status", "FAIL").upper())

    ws_res = result.get("workspace_result") or {}
    init_res = ws_res.get("initialize", {})
    print("   [Workspace] Init project  ->", init_res.get("status", "FAIL").upper())

    writes = ws_res.get("writes", [])
    writes_status = "SUCCESS" if all(w.get("result", {}).get("status") == "success" for w in writes) else "FAILED"
    print(f"   [Workspace] Write files   -> {writes_status} ({len(writes)} artifacts)")

    verif = result.get("verification") or {}
    syntax_res = verif.get("syntax", {})
    print("   [Verifier]  Syntax check  ->", syntax_res.get("status", "FAIL").upper())

    pytest_res = verif.get("pytest", {})
    print("   [Verifier]  Pytest check  ->", pytest_res.get("status", "FAIL").upper())

    print("\n==========================================")
    print(f"FINAL STATUS: {result.get('status', 'unknown').upper()}")
    print("==========================================\n")

    if pytest_res.get("stdout"):
        print("--- Pytest Output ---")
        print(pytest_res["stdout"].strip())
    if pytest_res.get("stderr") and pytest_res.get("status") != "passed":
        print("--- Pytest Error ---")
        print(pytest_res["stderr"].strip())

    out_path = Path(project_root) / "ForgeCore" / "paper_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nFull execution details saved to: {out_path}")


if __name__ == "__main__":
    target_pdf = sys.argv[1] if len(sys.argv) > 1 else "sample_cnn_paper.pdf"
    # Parse optional --dataset argument
    ds_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            ds_name = sys.argv[i + 1]
            break
    asyncio.run(run_paper_to_code(target_pdf, dataset_name=ds_name))
