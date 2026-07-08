import json

from openai import AsyncOpenAI
from pydantic import ValidationError

from paperforge_env import load_project_env

load_project_env()

from .generator import (
    normalize_python_content,
)
from .prompts import (
    REVIEW_IMPLEMENTOR_SYSTEM_PROMPT
)

from .schemas import (
    GeneratedFile,
    ImplementationPlan,
    ImplementationReview,
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


async def implement_review(
    spec: str,
    plan: dict,
    review: dict,
    artifacts: list[dict],
    error_context: str | None = None
) -> list[dict]:
    """
    Apply review feedback to fix generated
    artifacts that have issues.

    Returns only the artifacts that were
    changed to resolve review issues.
    """

    spec = spec.strip()

    if not spec:
        raise ValueError(
            "spec cannot be empty"
        )

    validated_plan = (
        ImplementationPlan.model_validate(
            plan
        )
    )

    validated_review = (
        ImplementationReview.model_validate(
            review
        )
    )

    validated_artifacts = [
        GeneratedFile.model_validate(
            artifact
        )
        for artifact in artifacts
    ]

    if not validated_artifacts:
        raise ValueError(
            "artifacts cannot be empty"
        )

    if validated_review.passed and not validated_review.issues:
        return []

    affected_paths = collect_affected_paths(
        review=validated_review
    )

    validate_affected_paths(
        affected_paths=affected_paths,
        artifacts=validated_artifacts
    )

    affected_artifacts = select_affected_artifacts(
        artifacts=validated_artifacts,
        affected_paths=affected_paths
    )

    payload = {
        "specification": spec,
        "implementation_plan":
            validated_plan.model_dump(),
        "review": validated_review.model_dump(),
        "affected_paths": sorted(
            affected_paths
        ),
        "affected_artifacts": [
            artifact.model_dump()
            for artifact
            in affected_artifacts
        ]
    }

    user_content = (
        "Fix the review issues "
        "in these artifacts:\n\n"
        + json.dumps(
            payload,
            indent=2,
            ensure_ascii=False
        )
        + "\n\nCRITICAL: You must respond ONLY with a valid JSON array matching the repaired artifacts schema (list of objects with keys 'path', 'content', 'language', 'dependencies_used', 'assumptions'). Do not write markdown text or explanations outside the JSON array. Start your response directly with '['."
    )
    if error_context:
        user_content += (
            f"\n\nCRITICAL: Your previous repair attempt failed validation with error:\n"
            f"{error_context}\n"
            "You MUST fix the exact syntax/format issue above. "
            "Use strict 4-space indentation after every 'def', 'class', 'if', 'for', 'while', 'with', and 'try'. "
            "Return only the affected files, preserving unchanged logic outside the required fix."
        )

    content = await stream_chat_with_retries(
        client=client,
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "system",
                "content":
                    REVIEW_IMPLEMENTOR_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_content
            }
        ],
        temperature=0.1,
        max_retries=5,
        base_delay=2.0,
        log_prefix="[ForgeCode] [Implementor]"
    )

    if not content:
        raise ValueError(
            "Implementor returned empty content"
        )

    raw_result = extract_json(
        content
    )

    fixed_artifacts = normalize_fixed_artifacts(
        raw_result=raw_result
    )

    if affected_paths and not fixed_artifacts:
        raise ValueError(
            "Implementor returned no fixed "
            "artifacts for a review containing "
            "affected files"
        )

    try:
        validated_fixed = [
            GeneratedFile.model_validate(
                artifact
            )
            for artifact in fixed_artifacts
        ]

    except ValidationError as error:
        raise ValueError(
            "Fixed artifacts failed validation: "
            f"{error}"
        ) from error

    validate_fixed_artifacts(
        fixed_artifacts=validated_fixed,
        original_artifacts=validated_artifacts,
        affected_paths=affected_paths
    )

    return [
        artifact.model_dump()
        for artifact in validated_fixed
    ]


def collect_affected_paths(
    review: ImplementationReview
) -> set[str]:
    """
    Collect every file path mentioned in
    review issues.
    """

    paths: set[str] = set()

    for issue in review.issues:
        paths.update(
            issue.affected_files
        )

    return paths


def validate_affected_paths(
    affected_paths: set[str],
    artifacts: list[GeneratedFile]
) -> None:
    """
    Ensure every affected path exists in
    the provided artifacts.
    """

    artifact_paths = {
        artifact.path
        for artifact in artifacts
    }

    unknown = (
        affected_paths
        - artifact_paths
    )

    if unknown:
        affected_paths.intersection_update(artifact_paths)
        if not affected_paths:
            affected_paths.update(artifact_paths)


def select_affected_artifacts(
    artifacts: list[GeneratedFile],
    affected_paths: set[str]
) -> list[GeneratedFile]:
    """
    Narrow the repair payload to the files that
    actually need edits.
    """

    if not affected_paths:
        return artifacts

    return [
        artifact
        for artifact in artifacts
        if artifact.path in affected_paths
    ]


def validate_fixed_artifacts(
    fixed_artifacts: list[GeneratedFile],
    original_artifacts: list[GeneratedFile],
    affected_paths: set[str]
) -> None:
    """
    Enforce that the implementor only touched
    affected artifacts and did not invent new
    paths.
    """

    original_paths = {
        artifact.path
        for artifact in original_artifacts
    }

    fixed_paths = [
        artifact.path
        for artifact in fixed_artifacts
    ]

    unknown_paths = (
        set(fixed_paths)
        - original_paths
    )

    if unknown_paths:
        raise ValueError(
            "Implementor returned unplanned "
            "paths: "
            f"{sorted(unknown_paths)}"
        )

    if len(fixed_paths) != len(
        set(fixed_paths)
    ):
        raise ValueError(
            "Duplicate fixed artifact paths "
            "detected"
        )

    unaffected_changed = (
        set(fixed_paths)
        - affected_paths
    )

    if unaffected_changed:
        raise ValueError(
            "Implementor changed unaffected "
            "artifacts: "
            f"{sorted(unaffected_changed)}"
        )

    validate_fixed_content_format(
        fixed_artifacts
    )


def validate_fixed_content_format(
    artifacts: list[GeneratedFile]
) -> None:
    """
    Perform deterministic format validation
    on fixed artifacts.
    """

    for artifact in artifacts:

        suffix = (
            artifact.path
            .lower()
            .rsplit(".", 1)[-1]
            if "." in artifact.path
            else ""
        )

        if suffix == "json":
            try:
                json.loads(
                    artifact.content
                )
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Fixed JSON content is "
                    f"invalid in {artifact.path}: "
                    f"{error}"
                ) from error

        if suffix == "py":
            try:
                compile(
                    artifact.content,
                    artifact.path,
                    "exec"
                )
            except SyntaxError as error:
                lines = artifact.content.splitlines()
                err_line = error.lineno or 1
                start_l = max(0, err_line - 3)
                end_l = min(len(lines), err_line + 2)
                snippet = "\n".join(
                    f"{i+1}: {lines[i]}"
                    for i in range(start_l, end_l)
                )
                raise ValueError(
                    "Fixed Python content has "
                    "invalid syntax in "
                    f"{artifact.path} on line {err_line}: {error.msg}\n"
                    f"Code snippet around line {err_line}:\n{snippet}"
                ) from error


def normalize_fixed_artifacts(
    raw_result
) -> list[dict]:
    """
    Coerce model output into a list of
    artifact dicts.

    Handles both a bare list and a wrapper
    dict with a 'fixed_artifacts' key.
    """

    items = []
    if isinstance(raw_result, list):
        items = raw_result
    elif isinstance(raw_result, dict):
        for key in ("fixed_artifacts", "artifacts", "files"):
            if isinstance(raw_result.get(key), list):
                items = raw_result[key]
                break

    normalized = []
    for item in items:
        if isinstance(item, dict):
            content = (
                item.get("content")
                or item.get("code")
                or item.get("source")
                or item.get("source_code")
                or item.get("file_content")
            )
            path = item.get("path") or item.get("filename")
            if path and content is not None:
                sfx = str(path).lower().rsplit(".", 1)[-1] if "." in str(path) else ""
                lang = item.get("language") or ("python" if sfx == "py" else ("json" if sfx == "json" else "text"))
                normalized_content = str(content)
                if str(lang).lower() == "python" or sfx == "py":
                    normalized_content = normalize_python_content(
                        normalized_content,
                        path=str(path)
                    )
                normalized.append({
                    "path": str(path),
                    "content": normalized_content,
                    "language": str(lang),
                    "dependencies_used": list(item.get("dependencies_used") or []),
                    "assumptions": list(item.get("assumptions") or [])
                })
            else:
                normalized.append(item)
        else:
            normalized.append(item)
    return normalized
