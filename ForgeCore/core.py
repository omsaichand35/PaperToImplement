from pathlib import Path
import re
from typing import Any

from codeforge.orchestrator import forge_implementation
from codeforge.implementor import implement_review
from router import EvidenceRouter, enrich_spec_with_evidence, enrich_spec_dict_with_evidence
from adapters.understanding_adapter import spec_from_understanding, spec_dict_from_understanding
from null_resolver import resolve_null_entities
import json


async def forge_project(
    spec: str | None = None,
    workspace: Any = None,
    knowledge: Any | None = None,
    research: Any | None = None,
    understanding: Any | None = None,
    pdf_path: str | None = None,
    project_name: str | None = None,
    max_review_rounds: int = 3,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Complete PaperForge V1 pipeline with Intelligent Router and PaperUnderstanding.

    [PDF Paper or Spec]
      -> PaperUnderstanding (if PDF provided: extracts structured implementation details)
      -> Null Entity Resolution (fills null/UNKNOWN facts)
      -> ForgeCore Router (Knowledge / Research triage)
      -> ForgeCode orchestration (with Grounded Evidence Bundle)
      -> semantic review
      -> workspace initialization
      -> artifact materialization
      -> syntax verification
      -> pytest verification
      -> post-failure diagnosis (if verification fails)
    """

    # ---------------------------------
    # 0. PaperUnderstanding Extraction
    # ---------------------------------
    print(f"\n[ForgeCore] Starting forge_project pipeline...", flush=True)
    understanding_result = None
    resolution_report = None
    pre_evidence = None
    target_pdf = pdf_path or (spec.strip() if (spec and spec.strip().lower().endswith(".pdf")) else None)

    if target_pdf and understanding:
        print(f"[ForgeCore] Step 0: Analyzing paper via PaperUnderstanding ({target_pdf})...", flush=True)
        spec_obj = await understanding.analyze(target_pdf)
        understanding_result = spec_obj.model_dump() if hasattr(spec_obj, "model_dump") else spec_obj

        # ---------------------------------
        # 0.5 Null Entity Resolution
        # ---------------------------------
        print("[ForgeCore] Step 0.5: Running Null Entity Resolution Engine...", flush=True)
        resolution_report = resolve_null_entities(
            spec_obj,
            dataset_name=dataset_name,
            research=research,
        )
        # Lossless path: keep the full ImplementationSpec as a structured dict
        spec_dict = spec_dict_from_understanding(spec_obj)

        # Attach router evidence as structured keys (not Markdown prose)
        pre_evidence = EvidenceRouter(
            knowledge=knowledge,
            research=research,
        ).route_pre_execution(spec_dict.get("paper_title") or "")
        grounded_spec_dict = enrich_spec_dict_with_evidence(spec_dict, pre_evidence)

        # Serialize to JSON string — forge_implementation still receives a str
        spec = json.dumps(grounded_spec_dict, indent=2, ensure_ascii=False, default=str)
        understanding_result = spec_dict  # keep for downstream reporting

    if not spec or not spec.strip():
        raise ValueError(
            "spec (or extracted PDF understanding) cannot be empty"
        )
    spec = spec.strip()

    # ---------------------------------
    # 1. Intelligent Pre-Execution Routing (text-spec path, for non-PDF inputs)
    # ---------------------------------
    print("[ForgeCore] Step 1: Running EvidenceRouter pre-execution triage...", flush=True)
    router = EvidenceRouter(
        knowledge=knowledge,
        research=research,
    )

    # Only run text-based routing when spec was NOT already enriched above (PDF path)
    if target_pdf and understanding:
        # Already enriched as dict; skip duplicate routing
        grounded_spec = spec
    else:
        pre_evidence = router.route_pre_execution(spec)
        grounded_spec = enrich_spec_with_evidence(spec, pre_evidence)

    # ---------------------------------
    # 2. Plan, generate, review, repair
    # ---------------------------------
    print("[ForgeCore] Step 2: Running ForgeCode implementation pipeline (planning, generation, review)...", flush=True)
    code_result = await forge_implementation(
        spec=grounded_spec,
        max_review_rounds=max_review_rounds,
    )

    code_status = code_result.get("status", "unknown")
    print(f"[ForgeCore] ForgeCode returned status: '{code_status}'. Proceeding to write artifacts to disk...", flush=True)
    if code_status not in ("passed", "review_failed"):
        print(f"[ForgeCore] Warning: Unexpected code_result status '{code_status}'. Will still attempt to materialize artifacts.", flush=True)

    plan = code_result["plan"]
    artifacts = code_result["artifacts"]

    resolved_project_name = (
        project_name
        or plan["project_name"]
    )

    # ---------------------------------
    # 3. Initialize real workspace
    # ---------------------------------
    print(f"[ForgeCore] Step 3: Initializing real workspace for project '{resolved_project_name}'...", flush=True)
    init_result = workspace.initialize(
        resolved_project_name
    )

    if init_result.get("status") != "success":
        print(f"[ForgeCore] Workspace initialization failed: {init_result}", flush=True)
        return {
            "status": "workspace_initialization_failed",
            "stage": "workspace_init",
            "routing_evidence": pre_evidence,
            "code_result": code_result,
            "workspace_result": {
                "initialize": init_result
            },
            "verification": None,
        }

    # ---------------------------------
    # 4. Materialize reviewed artifacts
    # ---------------------------------
    print(f"[ForgeCore] Step 4: Materializing {len(plan['implementation_order'])} artifacts to disk...", flush=True)
    write_results = []

    artifacts_by_path = {
        artifact["path"]: artifact
        for artifact in artifacts
    }

    for path in plan["implementation_order"]:

        artifact = artifacts_by_path.get(
            path
        )

        if artifact is None:
            print(f"[ForgeCore] Missing artifact for planned path: {path}", flush=True)
            return {
                "status": "artifact_missing",
                "stage": "workspace_write",
                "missing_path": path,
                "routing_evidence": pre_evidence,
                "code_result": code_result,
                "workspace_result": {
                    "initialize": init_result,
                    "writes": write_results,
                },
                "verification": None,
            }

        write_result = workspace.write(
            resolved_project_name,
            path,
            artifact["content"],
        )

        write_results.append({
            "path": path,
            "result": write_result,
        })

        if write_result.get("status") != "success":
            print(f"[ForgeCore] Failed writing file '{path}': {write_result}", flush=True)
            return {
                "status": "workspace_write_failed",
                "stage": "workspace_write",
                "failed_path": path,
                "routing_evidence": pre_evidence,
                "code_result": code_result,
                "workspace_result": {
                    "initialize": init_result,
                    "writes": write_results,
                },
                "verification": None,
            }

    # ---------------------------------
    # 5 & 6. Closed-Loop Execution & Self-Healing (Pillar 5)
    # ---------------------------------
    max_runtime_retries = 2
    syntax_result = None
    pytest_result = None
    diag_evidence = None

    for attempt in range(max_runtime_retries + 1):
        if attempt > 0:
            print(f"\n[ForgeCore] --- Self-Healing Repair Cycle ({attempt}/{max_runtime_retries}) ---", flush=True)

        print(f"[ForgeCore] Step 5: Running syntax verification across project '{resolved_project_name}'...", flush=True)
        syntax_result = workspace.verify_syntax(
            resolved_project_name
        )

        if syntax_result.get("status") != "passed":
            error_output = syntax_result.get("stderr") or syntax_result.get("stdout") or "Syntax error detected."
            print(f"[ForgeCore] Syntax verification failed! Running router failure diagnosis...", flush=True)
            diag_evidence = router.route_post_failure(
                failed_stage="syntax",
                error_output=error_output,
            )
            if attempt >= max_runtime_retries:
                print("[ForgeCore] Max runtime repair attempts exhausted on syntax verification.", flush=True)
                return {
                    "status": "syntax_verification_failed",
                    "stage": "syntax_verification",
                    "routing_evidence": pre_evidence,
                    "diagnostic_evidence": diag_evidence,
                    "code_result": code_result,
                    "workspace_result": {
                        "initialize": init_result,
                        "writes": write_results,
                    },
                    "verification": {
                        "syntax": syntax_result,
                        "pytest": None,
                    },
                }
            await _run_self_repair(
                failed_stage="syntax",
                error_output=error_output,
                diag_evidence=diag_evidence,
                grounded_spec=grounded_spec,
                plan=plan,
                artifacts_by_path=artifacts_by_path,
                resolved_project_name=resolved_project_name,
                workspace=workspace,
                write_results=write_results,
            )
            code_result["artifacts"] = list(artifacts_by_path.values())
            continue

        print(f"[ForgeCore] Step 6: Running runtime pytest verification across project '{resolved_project_name}'...", flush=True)
        pytest_result = workspace.verify_tests(
            resolved_project_name
        )

        if pytest_result.get("status") != "passed":
            error_output = pytest_result.get("stderr") or pytest_result.get("stdout") or "Pytest runtime error detected."
            print(f"[ForgeCore] Pytest runtime verification failed! Running router failure diagnosis...", flush=True)
            diag_evidence = router.route_post_failure(
                failed_stage="pytest",
                error_output=error_output,
            )
            if attempt >= max_runtime_retries:
                print("[ForgeCore] Max runtime repair attempts exhausted on pytest verification.", flush=True)
                return {
                    "status": "pytest_verification_failed",
                    "stage": "pytest_verification",
                    "routing_evidence": pre_evidence,
                    "diagnostic_evidence": diag_evidence,
                    "code_result": code_result,
                    "workspace_result": {
                        "initialize": init_result,
                        "writes": write_results,
                    },
                    "verification": {
                        "syntax": syntax_result,
                        "pytest": pytest_result,
                    },
                }
            await _run_self_repair(
                failed_stage="pytest",
                error_output=error_output,
                diag_evidence=diag_evidence,
                grounded_spec=grounded_spec,
                plan=plan,
                artifacts_by_path=artifacts_by_path,
                resolved_project_name=resolved_project_name,
                workspace=workspace,
                write_results=write_results,
            )
            code_result["artifacts"] = list(artifacts_by_path.values())
            continue

        print("[ForgeCore] Syntax and Pytest verification PASSED cleanly!", flush=True)
        break

    # ---------------------------------
    # 7. Complete
    # ---------------------------------
    final_status = "passed" if code_result["status"] == "passed" else "semantic_review_failed"
    print(f"[ForgeCore] Step 7: Pipeline complete with status '{final_status}' for project '{resolved_project_name}'.\n==========================================\n", flush=True)
    return {
        "status": final_status,
        "stage": "complete",
        "project_name": resolved_project_name,
        "understanding_result": understanding_result,
        "resolution_report": resolution_report,
        "routing_evidence": pre_evidence,
        "code_result": code_result,
        "workspace_result": {
            "initialize": init_result,
            "writes": write_results,
        },
        "verification": {
            "syntax": syntax_result,
            "pytest": pytest_result,
        },
    }


def _format_diagnostic_evidence(
    diag_evidence: dict | None
) -> str:
    enriched = ""
    if diag_evidence and diag_evidence.get("diagnostic_knowledge"):
        enriched += "\nOfficial API Truth Diagnostic:\n"
        for item in diag_evidence["diagnostic_knowledge"]:
            enriched += (
                f"- {item.get('symbol') or item.get('title')}: "
                f"{item.get('signature') or ''} "
                f"{item.get('description') or ''}\n"
            )
    if diag_evidence and diag_evidence.get("diagnostic_research"):
        enriched += "\nResearch Diagnostic:\n"
        for item in diag_evidence["diagnostic_research"]:
            enriched += f"- {item.get('title')}\n"
    return enriched


def _paths_mentioned_in_output(
    error_output: str,
    artifacts_by_path: dict,
) -> list[str]:
    affected_files = []
    for path in artifacts_by_path.keys():
        if path in error_output or Path(path).name in error_output:
            affected_files.append(path)
    return affected_files


def _missing_symbol_issue(
    symbol: str,
    error_output: str,
    artifacts_by_path: dict,
) -> dict:
    affected = _paths_mentioned_in_output(
        error_output,
        artifacts_by_path,
    )

    for path, artifact in artifacts_by_path.items():
        if path.endswith(".py") and symbol in artifact.get("content", ""):
            if path not in affected:
                affected.append(path)

    if not affected:
        affected = [
            path
            for path in artifacts_by_path
            if path.endswith(".py")
        ]

    return {
        "severity": "critical",
        "category": "interface_mismatch",
        "message": (
            f"Runtime verification failed because symbol '{symbol}' is referenced "
            "but not defined or imported in the execution path."
        ),
        "affected_files": affected,
        "evidence": [error_output[:1000]],
        "recommendation": (
            f"Define and export '{symbol}' in the appropriate module or update "
            "call sites/imports so all references resolve. Keep tests aligned "
            "with the actual public API."
        ),
    }


def _attribute_error_issue(
    module: str,
    attribute: str,
    error_output: str,
    artifacts_by_path: dict,
) -> dict:
    affected = _paths_mentioned_in_output(
        error_output,
        artifacts_by_path,
    )
    if not affected:
        affected = [
            path
            for path in artifacts_by_path
            if path.endswith(".py")
        ]

    return {
        "severity": "critical",
        "category": "interface_mismatch",
        "message": (
            f"Runtime verification failed because module '{module}' does not "
            f"provide attribute '{attribute}'."
        ),
        "affected_files": affected,
        "evidence": [error_output[:1000]],
        "recommendation": (
            "Remove or guard the unavailable API call. Use only APIs available "
            "in installed dependencies, add an optional fallback implementation, "
            "or record the external component as unresolved without crashing."
        ),
    }


def _generic_runtime_issue(
    failed_stage: str,
    enriched_error: str,
    error_output: str,
    artifacts_by_path: dict,
) -> dict:
    affected = _paths_mentioned_in_output(
        error_output,
        artifacts_by_path,
    )
    if not affected:
        affected = [
            path
            for path in artifacts_by_path.keys()
            if path.endswith(".py")
        ]
    if not affected:
        affected = list(artifacts_by_path.keys())

    return {
        "severity": "critical",
        "category": "other",
        "message": enriched_error[:2500],
        "affected_files": affected,
        "evidence": [error_output[:1000]],
        "recommendation": (
            "Fix the smallest set of affected files needed to pass verification. "
            "Preserve the architecture contract and avoid broad rewrites."
        ),
    }


def _build_runtime_repair_review(
    failed_stage: str,
    error_output: str,
    enriched_error: str,
    artifacts_by_path: dict,
) -> dict:
    issues = []

    for module, attribute in re.findall(
        r"AttributeError:\s+module '([^']+)' has no attribute '([^']+)'",
        error_output,
    ):
        issues.append(
            _attribute_error_issue(
                module=module,
                attribute=attribute,
                error_output=error_output,
                artifacts_by_path=artifacts_by_path,
            )
        )

    for symbol in re.findall(
        r"NameError:\s+name '([^']+)' is not defined",
        error_output,
    ):
        issues.append(
            _missing_symbol_issue(
                symbol=symbol,
                error_output=error_output,
                artifacts_by_path=artifacts_by_path,
            )
        )

    if not issues:
        issues.append(
            _generic_runtime_issue(
                failed_stage=failed_stage,
                enriched_error=enriched_error,
                error_output=error_output,
                artifacts_by_path=artifacts_by_path,
            )
        )

    affected_paths = sorted({
        path
        for issue in issues
        for path in issue["affected_files"]
    })

    return {
        "passed": False,
        "summary": (
            f"Runtime/Syntax verification failed during stage '{failed_stage}'. "
            "Self-healing repair required."
        ),
        "checked_files": list(artifacts_by_path.keys()),
        "missing_requirements": [],
        "invented_details": [],
        "cross_file_inconsistencies": [],
        "issues": issues,
        "_affected_paths": affected_paths,
    }


async def _run_self_repair(
    failed_stage: str,
    error_output: str,
    diag_evidence: dict,
    grounded_spec: str,
    plan: dict,
    artifacts_by_path: dict,
    resolved_project_name: str,
    workspace: Any,
    write_results: list,
):
    print(f"[ForgeCore] [Self-Healing] Injecting '{failed_stage}' traceback into implementor for auto-repair...", flush=True)
    
    enriched_error = f"Verification failed during '{failed_stage}' with traceback:\n{error_output}\n"
    enriched_error += _format_diagnostic_evidence(
        diag_evidence
    )
    enriched_error += (
        "\nSelf-healing constraints:\n"
        "- Repair only files named in affected_files.\n"
        "- Do not call unavailable package APIs; guard optional dependencies or provide local fallbacks.\n"
        "- If a test references a symbol, either export that symbol or correct the test import/call site.\n"
        "- Preserve architecture-contract requirements while making the project executable.\n"
    )

    diagnostic_review = _build_runtime_repair_review(
        failed_stage=failed_stage,
        error_output=error_output,
        enriched_error=enriched_error,
        artifacts_by_path=artifacts_by_path,
    )
    affected_paths = diagnostic_review.pop(
        "_affected_paths",
        []
    )
    print(
        f"[ForgeCore] [Self-Healing] Targeted affected files: {affected_paths}",
        flush=True
    )

    try:
        raw_fixed = await implement_review(
            spec=grounded_spec,
            plan=plan,
            review=diagnostic_review,
            artifacts=list(artifacts_by_path.values()),
            error_context=enriched_error[:4000],
        )
        if not raw_fixed:
            print("[ForgeCore] [Self-Healing] Implementor returned no repairs.", flush=True)
            return
        for fa in raw_fixed:
            artifacts_by_path[fa["path"]] = fa
            print(f"[ForgeCore] [Self-Healing] Repaired artifact received: {fa['path']}. Overwriting disk...", flush=True)
            res = workspace.write(resolved_project_name, fa["path"], fa["content"])
            write_results.append({"path": fa["path"], "result": res, "repaired": True})
    except Exception as exc:
        print(f"[ForgeCore] [Self-Healing] Repair attempt encountered error: {exc}", flush=True)
