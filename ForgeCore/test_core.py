import asyncio
import json
import sys
from pathlib import Path

# Add paths so ForgeCode, ForgeWorkspace, ForgeKnowledge, ForgeResearch, and ForgeCore are importable
project_root = r"C:\Users\omsai\PycharmProjects\PaperForge"
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeCode")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeWorkspace")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeKnowledge")
    sys.path.insert(0, r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeResearch")

from core import forge_project
from adapters.workspace_adapter import WorkspaceAdapter
from adapters.knowledge_adapter import KnowledgeAdapter
from adapters.research_adapter import ResearchAdapter

# Import REAL ForgeWorkspace functions
from workspace.project import initialize_project
from workspace.files import write_project_file
from workspace.verifer import verify_project

# Import REAL ForgeKnowledge functions
from retrieval.search import search_documents, find_exact_symbol

# Import REAL ForgeResearch functions
from research.discovery import find_research_papers
from research.inspector import inspect_research_page

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

SPEC = """
Build a PyTorch 1D CNN classifier.

Input tensor shape:
(batch, 2, 1024)

Architecture:
- Conv1d: 2 to 32 channels, kernel size 7
- BatchNorm1d with 32 features
- ReLU
- MaxPool1d with kernel size 2
- Conv1d: 32 to 64 channels, kernel size 5
- BatchNorm1d with 64 features
- ReLU
- AdaptiveAvgPool1d output size 1
- Linear layer: 64 to 5 classes

Requirements:
- model implementation
- model configuration
- forward-pass shape test

Do not add training code.
Do not add dataset code.
"""


async def main():
    result = await forge_project(
        spec=SPEC,
        workspace=workspace,
        knowledge=knowledge,
        research=research,
        max_review_rounds=3,
    )

    print("\n==========================================")
    print("         INTELLIGENT ROUTER RESULT        ")
    print("==========================================\n")

    routing = result.get("routing_evidence", {})
    print("Evidence Router Triage:")
    print(f"  - Routed to Knowledge: {routing.get('routed_to_knowledge', False)} ({len(routing.get('knowledge_evidence', []))} items)")
    print(f"  - Routed to Research: {routing.get('routed_to_research', False)} ({len(routing.get('research_evidence', []))} items)")
    print("\n==========================================\n")

    code_res = result.get("code_result", {})
    print("ForgeCode semantic review")
    print(f"        {code_res.get('status', 'FAIL').upper()}")
    print("          v")

    ws_res = result.get("workspace_result") or {}
    init_res = ws_res.get("initialize", {})
    print("Workspace initialized")
    print(f"        {init_res.get('status', 'FAIL').upper()}")
    print("          v")

    writes = ws_res.get("writes", [])
    writes_status = "SUCCESS" if all(w.get("result", {}).get("status") == "success" for w in writes) else "FAILED"
    print(f"{len(writes)} artifacts written")
    print(f"        {writes_status}")
    print("          v")

    verif = result.get("verification") or {}
    syntax_res = verif.get("syntax", {})
    print("Syntax verification")
    print(f"        {syntax_res.get('status', 'FAIL').upper()}")
    print("          v")

    pytest_res = verif.get("pytest", {})
    print("pytest")
    print(f"        {pytest_res.get('status', 'FAIL').upper()}")
    print("          v")

    print(f"status = {result.get('status')}")
    print(f"stage = {result.get('stage')}")
    print("\n==========================================")

    if pytest_res.get("stdout"):
        print("\n--- Pytest Output ---")
        print(pytest_res["stdout"].strip())

    out_path = Path(r"C:\Users\omsai\PycharmProjects\PaperForge\ForgeCore\result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nFull result saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
