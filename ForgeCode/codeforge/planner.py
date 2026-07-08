import json
from pathlib import Path
import sys

from openai import AsyncOpenAI

from paperforge_env import load_project_env

load_project_env()

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    from codeforge.prompts import (
        PLANNER_SYSTEM_PROMPT
    )

    from codeforge.schemas import (
        ImplementationPlan,
        extract_json,
        stream_chat_with_retries,
    )

else:
    from .prompts import (
        PLANNER_SYSTEM_PROMPT
    )

    from .schemas import (
        ImplementationPlan,
        extract_json,
        stream_chat_with_retries,
    )


client = AsyncOpenAI(
    base_url=(
        "https://integrate.api.nvidia.com/v1"
    ),
    max_retries=3,
    timeout=300.0,
)


async def create_implementation_plan(
    spec: str
) -> dict:
    """
    Convert an implementation specification
    into a validated structured plan.
    """

    spec = spec.strip()

    if not spec:
        raise ValueError(
            "spec cannot be empty"
        )

    content = await stream_chat_with_retries(
        client=client,
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "system",
                "content":
                    PLANNER_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Create an implementation "
                    "plan for this specification:\n\n"
                    f"{spec}\n\n"
                    "CRITICAL: You must respond ONLY with a valid JSON object matching the Implementation Plan schema. Do not write markdown headings, text, or explanations outside the JSON object. Start your response directly with '{'."
                )
            }
        ],
        temperature=0.1,
        max_retries=5,
        base_delay=2.0,
        log_prefix="[ForgeCode] [Planner]"
    )

    if not content:
        raise ValueError(
            "Planner returned empty content"
        )

    raw_plan = extract_json(
        content
    )

    normalized_plan = normalize_plan(
        raw_plan=raw_plan,
        spec=spec
    )

    validated_plan = (
        ImplementationPlan
        .model_validate(
            normalized_plan
        )
    )

    return (
        validated_plan
        .model_dump()
    )



def normalize_plan(
    raw_plan: dict,
    spec: str
) -> dict:
    """
    Coerce model output into the schema expected by
    ImplementationPlan.

    The planner may emit nested requirement buckets or
    file maps, so this function derives missing top-level
    fields and reshapes file data into a list.
    """

    plan = dict(
        raw_plan
    )

    explicit_requirements = plan.pop(
        "explicit_requirements",
        None
    )

    if isinstance(
        explicit_requirements,
        dict
    ) and not plan.get(
        "summary"
    ):
        plan["summary"] = summarize_requirements(
            explicit_requirements
        )

    plan["project_name"] = plan.get(
        "project_name"
    ) or infer_project_name(
        spec
    )

    plan["framework"] = plan.get(
        "framework"
    ) or infer_framework(
        spec
    )

    plan["task_type"] = plan.get(
        "task_type"
    ) or infer_task_type(
        spec
    )

    plan["summary"] = plan.get(
        "summary"
    ) or build_summary(
        spec
    )

    files = (
        plan.get("files")
        or plan.get("planned_files")
        or plan.get("project_files")
        or plan.get("file_list")
        or plan.get("artifacts")
    )

    plan["files"] = normalize_files(
        files
    )

    if not plan["files"] and plan.get("implementation_order"):
        derived = []
        order = normalize_string_list(plan.get("implementation_order"))
        for idx, path in enumerate(order):
            derived.append({
                "path": path,
                "purpose": f"Implementation file for {path}",
                "responsibilities": [f"Implement requirements for {path}"],
                "depends_on": order[:idx]
            })
        plan["files"] = derived

    if not plan["files"]:
        raise ValueError(
            "Planner returned no valid "
            "planned files"
        )

    valid_paths = {f["path"] for f in plan["files"] if f.get("path")}
    for f in plan["files"]:
        clean_deps = []
        for dep in f.get("depends_on", []):
            if dep in valid_paths and dep != f.get("path"):
                clean_deps.append(dep)
            else:
                for vp in valid_paths:
                    if (vp.endswith("/" + dep) or vp == dep) and vp != f.get("path"):
                        if vp not in clean_deps:
                            clean_deps.append(vp)
                        break
        f["depends_on"] = clean_deps

    plan["implementation_order"] = normalize_implementation_order(
        plan.get("implementation_order"),
        plan["files"]
    )

    plan["dependencies"] = normalize_string_list(
        plan.get("dependencies")
    )

    plan["assumptions"] = normalize_string_list(
        plan.get("assumptions")
    )

    plan["unresolved_questions"] = normalize_string_list(
        plan.get("unresolved_questions")
    )

    return plan


def infer_project_name(
    spec: str
) -> str:

    text = spec.strip().lower()

    if "classifier" in text:
        return "classifier_project"

    if "regression" in text:
        return "regression_project"

    if "segmentation" in text:
        return "segmentation_project"

    return "implementation_project"


def infer_framework(
    spec: str
) -> str:

    text = spec.lower()

    if "pytorch" in text:
        return "pytorch"

    if "tensorflow" in text:
        return "tensorflow"

    return "unknown"


def infer_task_type(
    spec: str
) -> str:

    text = spec.lower()

    if "1d" in text and (
        "classifier" in text
        or "classification" in text
    ):
        return "1d_classification"

    if "classifier" in text or "classification" in text:
        return "classification"

    if "regression" in text:
        return "regression"

    if "segmentation" in text:
        return "segmentation"

    return "implementation"


def build_summary(
    spec: str
) -> str:

    cleaned = " ".join(
        spec.strip().split()
    )

    if len(cleaned) <= 240:
        return cleaned

    return cleaned[:237].rstrip() + "..."


def summarize_requirements(
    explicit_requirements: dict
) -> str:

    items = []

    for key, value in explicit_requirements.items():

        if isinstance(value, str) and value.strip():
            items.append(
                f"{key}: {value.strip()}"
            )
        else:
            items.append(
                str(key)
            )

    return "; ".join(
        items
    )


def normalize_string_list(
    value
) -> list[str]:

    if value is None:
        return []

    if isinstance(
        value,
        str
    ):
        value = [value]

    if not isinstance(
        value,
        list
    ):
        return []

    normalized = []

    for item in value:

        if item is None:
            continue

        text = str(
            item
        ).strip()

        if text:
            normalized.append(
                text
            )

    return normalized


def normalize_files(
    files
) -> list[dict]:

    if files is None:
        return []

    if isinstance(
        files,
        list
    ):
        normalized = []

        for item in files:

            if isinstance(
                item,
                dict
            ):
                normalized.append(
                    normalize_file_entry(
                        item
                    )
                )

        return normalized

    if isinstance(
        files,
        dict
    ):
        normalized = []

        for path, details in files.items():
            normalized.append(
                normalize_file_entry(
                    details,
                    fallback_path=str(path)
                )
            )

        return normalized

    return []


def normalize_file_entry(
    details,
    fallback_path: str | None = None
) -> dict:

    if isinstance(
        details,
        str
    ):
        return {
            "path": fallback_path or "planned_file.py",
            "purpose": details.strip(),
            "responsibilities": [],
            "depends_on": []
        }

    if not isinstance(
        details,
        dict
    ):
        return {
            "path": fallback_path or "planned_file.py",
            "purpose": "Planned implementation file",
            "responsibilities": [],
            "depends_on": []
        }

    path = details.get(
        "path"
    ) or fallback_path or "planned_file.py"

    purpose = details.get(
        "purpose"
    ) or details.get(
        "description"
    ) or details.get(
        "summary"
    ) or "Planned implementation file"

    responsibilities = normalize_string_list(
        details.get("responsibilities")
    )

    depends_on = normalize_string_list(
        details.get("depends_on")
    )

    return {
        "path": str(
            path
        ).strip(),
        "purpose": str(
            purpose
        ).strip(),
        "responsibilities": responsibilities,
        "depends_on": depends_on
    }


def normalize_implementation_order(
    implementation_order,
    files: list[dict]
) -> list[str]:

    file_map = {
        f["path"]: set(f.get("depends_on", []))
        for f in files
        if f.get("path")
    }

    ordered = []
    visited = set()
    visiting = set()

    def visit(path):
        if path in visiting:
            return
        if path not in visited:
            visiting.add(path)
            for dep in file_map.get(path, []):
                if dep in file_map:
                    visit(dep)
            visiting.remove(path)
            visited.add(path)
            ordered.append(path)

    suggested = normalize_string_list(
        implementation_order
    ) or list(file_map.keys())

    for path in suggested:
        if path in file_map:
            visit(path)

    for path in file_map:
        visit(path)

    return ordered


