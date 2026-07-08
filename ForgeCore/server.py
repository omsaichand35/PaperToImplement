"""
PaperForge API Server
Drag & Drop PDF -> Implementation pipeline via FastAPI + SSE
"""
import asyncio
import json
import os
import sys
import shutil
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

# ──────────────────────────────────────────
# Sys-path bootstrap
# ──────────────────────────────────────────
PROJECT_ROOT = r"C:\Users\omsai\PycharmProjects\PaperForge"
for _p in [
    PROJECT_ROOT,
    rf"{PROJECT_ROOT}\ForgeCode",
    rf"{PROJECT_ROOT}\ForgeWorkspace",
    rf"{PROJECT_ROOT}\ForgeKnowledge",
    rf"{PROJECT_ROOT}\ForgeResearch",
    rf"{PROJECT_ROOT}\PaperUnderstanding",
    rf"{PROJECT_ROOT}\ForgeCore",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import forge_project
from adapters.workspace_adapter import WorkspaceAdapter
from adapters.knowledge_adapter import KnowledgeAdapter
from adapters.research_adapter import ResearchAdapter
from adapters.understanding_adapter import UnderstandingAdapter

from workspace.project import initialize_project
from workspace.files import write_project_file
from workspace.verifer import verify_project
from retrieval.search import search_documents, find_exact_symbol
from research.discovery import find_research_papers
from research.inspector import inspect_research_page
from client import analyze_paper

# ──────────────────────────────────────────
# Adapters
# ──────────────────────────────────────────
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

PAPERS_DIR = Path(PROJECT_ROOT) / "PaperUnderstanding" / "papers"
PAPERS_DIR.mkdir(parents=True, exist_ok=True)

WORKSPACE_PROJECTS = Path(PROJECT_ROOT) / "ForgeWorkspace" / "projects"
STATIC_DIR = Path(__file__).parent / "static"

# ──────────────────────────────────────────
# App
# ──────────────────────────────────────────
app = FastAPI(title="PaperForge", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/", response_class=HTMLResponse)
async def root():
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/forge")
async def forge_stream(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    save_path = PAPERS_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    async def pipeline_stream() -> AsyncGenerator[str, None]:
        try:
            yield sse_event({"stage": "upload", "status": "ok",
                             "message": f"Received: {file.filename}"})

            yield sse_event({"stage": "understanding", "status": "running",
                             "message": "Extracting implementation details from paper..."})

            task = asyncio.create_task(
                forge_project(
                    pdf_path=str(save_path),
                    workspace=workspace,
                    knowledge=knowledge,
                    research=research,
                    understanding=understanding,
                    max_review_rounds=3,
                )
            )

            elapsed = 0
            while not task.done():
                await asyncio.sleep(4)
                elapsed += 4
                if not task.done():
                    yield sse_event({
                        "stage": "understanding",
                        "status": "running",
                        "message": f"AI agents forging implementation ({elapsed}s elapsed)..."
                    })

            result = await task

            under = result.get("understanding_result") or {}
            yield sse_event({
                "stage": "understanding", "status": "ok",
                "message": "Paper understood",
                "detail": {
                    "title": under.get("paper_title"),
                    "task": under.get("task"),
                    "components": len(under.get("model_components", [])),
                }
            })

            routing = result.get("routing_evidence", {})
            yield sse_event({
                "stage": "routing", "status": "ok",
                "message": "Evidence bundle grounded",
                "detail": {
                    "knowledge_items": len(routing.get("knowledge_evidence", [])),
                    "research_items": len(routing.get("research_evidence", [])),
                }
            })

            code_res = result.get("code_result", {})
            yield sse_event({
                "stage": "forge_code", "status": code_res.get("status", "failed"),
                "message": "Implementation plan & artifacts generated",
                "detail": {
                    "review_rounds": code_res.get("review_rounds"),
                    "files": [f["path"] for f in code_res.get("plan", {}).get("files", [])]
                }
            })

            ws = result.get("workspace_result") or {}
            writes = ws.get("writes", [])
            yield sse_event({
                "stage": "workspace", "status": "ok" if writes else "failed",
                "message": f"{len(writes)} artifact(s) written to disk",
                "detail": {"files": [w["path"] for w in writes]}
            })

            verif = result.get("verification") or {}
            syntax = verif.get("syntax", {})
            pytest_r = verif.get("pytest", {})
            yield sse_event({
                "stage": "verification", "status": pytest_r.get("status", "failed"),
                "message": "Syntax + pytest verification complete",
                "detail": {
                    "syntax": syntax.get("status"),
                    "pytest": pytest_r.get("status"),
                    "pytest_output": pytest_r.get("stdout", ""),
                    "pytest_error": pytest_r.get("stderr", ""),
                }
            })

            project_name = result.get("project_name") or ""
            generated_files = {}
            project_dir = WORKSPACE_PROJECTS / project_name
            if project_dir.exists():
                for py_file in sorted(project_dir.rglob("*.py")):
                    rel = str(py_file.relative_to(project_dir))
                    try:
                        generated_files[rel] = py_file.read_text(encoding="utf-8")
                    except Exception:
                        generated_files[rel] = "# Could not read file"

            yield sse_event({
                "stage": "complete", "status": result.get("status", "unknown"),
                "message": "Pipeline complete",
                "detail": {
                    "project_name": project_name,
                    "final_status": result.get("status"),
                    "generated_files": generated_files,
                }
            })

        except Exception as exc:
            yield sse_event({"stage": "error", "status": "error", "message": str(exc)})

    return StreamingResponse(
        pipeline_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/projects")
async def list_projects():
    if not WORKSPACE_PROJECTS.exists():
        return JSONResponse({"projects": []})
    projects = [d.name for d in WORKSPACE_PROJECTS.iterdir() if d.is_dir()]
    return JSONResponse({"projects": sorted(projects)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
