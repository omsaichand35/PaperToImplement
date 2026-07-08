import json
import os

from openai import AsyncOpenAI
from pydantic import ValidationError

from paperforge_env import load_project_env

load_project_env()

from .prompts import (
    IMPLEMENTATION_REVIEW_SYSTEM_PROMPT
)

from .contract import (
    contract_issues,
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


async def review_implementation(
    spec: str,
    plan: dict,
    artifacts: list[dict]
) -> dict:
    """
    Review generated artifacts against
    the original spec and validated plan.
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

    validate_artifact_set(
        plan=validated_plan,
        artifacts=validated_artifacts
    )

    payload = {
        "specification": spec,
        "implementation_plan":
            validated_plan.model_dump(),
        "generated_artifacts": [
            artifact.model_dump()
            for artifact
            in validated_artifacts
        ]
    }

    content = await stream_chat_with_retries(
        client=client,
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {
                "role": "system",
                "content":
                    IMPLEMENTATION_REVIEW_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    "Review this implementation "
                    "evidence:\n\n"
                    + json.dumps(
                        payload,
                        indent=2,
                        ensure_ascii=False
                    )
                    + "\n\nCRITICAL: You must respond ONLY with a valid JSON object matching the Implementation Review schema. Do not write bullet points, text, or markdown outside the JSON object. Start your response directly with '{'."
                )
            }
        ],
        temperature=0.1,
        max_retries=5,
        base_delay=2.0,
        log_prefix="[ForgeCode] [Reviewer]"
    )

    if not content:
        raise ValueError(
            "Reviewer returned empty content"
        )

    raw_review = extract_json(
        content
    )

    try:
        review = (
            ImplementationReview
            .model_validate(
                raw_review
            )
        )

    except ValidationError as error:
        raise ValueError(
            "Implementation review failed "
            f"validation: {error}"
        ) from error

    validate_review_evidence(
        review=review,
        plan=validated_plan,
        artifacts=validated_artifacts
    )

    apply_contract_issues(
        review=review,
        spec=spec,
        artifacts=validated_artifacts,
    )

    return review.model_dump()


def validate_artifact_set(
    plan: ImplementationPlan,
    artifacts: list[GeneratedFile]
) -> None:
    """
    Validate artifact paths before asking
    the LLM to review them.
    """

    planned_paths = {
        file.path
        for file in plan.files
    }

    artifact_paths = [
        artifact.path
        for artifact in artifacts
    ]

    unknown_paths = (
        set(artifact_paths)
        - planned_paths
    )

    if unknown_paths:
        raise ValueError(
            "Artifacts contain unplanned paths: "
            f"{sorted(unknown_paths)}"
        )

    if len(artifact_paths) != len(
        set(artifact_paths)
    ):
        raise ValueError(
            "Duplicate artifact paths detected"
        )


def validate_review_evidence(
    review: ImplementationReview,
    plan: ImplementationPlan,
    artifacts: list[GeneratedFile]
) -> None:
    """
    Prevent the reviewer from referencing
    invented file paths.
    """

    allowed_paths = {
        file.path
        for file in plan.files
    }

    allowed_paths.update(
        artifact.path
        for artifact in artifacts
    )

    actual_artifact_paths = {
        artifact.path
        for artifact in artifacts
    }

    # Normalize review paths to match actual_artifact_paths where possible
    def normalize_review_path(p: str) -> str:
        if p in actual_artifact_paths:
            return p
        if f"{p}.py" in actual_artifact_paths:
            return f"{p}.py"
        candidates = []
        p_norm = p.replace("\\", "/").lower()
        for path in actual_artifact_paths:
            path_norm = path.replace("\\", "/").lower()
            stem = path_norm.rsplit(".", 1)[0]
            if stem == p_norm or stem.endswith("/" + p_norm):
                candidates.append(path)
                continue
            basename = path_norm.split("/")[-1]
            basename_stem = basename.rsplit(".", 1)[0]
            if basename == p_norm or basename_stem == p_norm:
                candidates.append(path)
        if len(candidates) == 1:
            return candidates[0]
        return p

    review.checked_files = [
        f for f in (normalize_review_path(f) for f in review.checked_files)
        if f in actual_artifact_paths
    ]
    if not review.checked_files:
        review.checked_files = sorted(actual_artifact_paths)

    for issue in review.issues:
        valid_affected = [
            f for f in (normalize_review_path(f) for f in issue.affected_files)
            if f in actual_artifact_paths
        ]
        issue.affected_files = valid_affected


def apply_contract_issues(
    review: ImplementationReview,
    spec: str,
    artifacts: list[GeneratedFile],
) -> None:
    """
    Add deterministic paper-to-code contract failures
    that the LLM reviewer may miss.
    """

    existing = {
        (
            issue.category,
            issue.message,
            tuple(issue.affected_files),
        )
        for issue in review.issues
    }

    for issue in contract_issues(
        spec=spec,
        artifacts=artifacts,
    ):
        key = (
            issue.category,
            issue.message,
            tuple(issue.affected_files),
        )
        if key not in existing:
            review.issues.append(issue)
            existing.add(key)

    if any(
        issue.severity in {"critical", "error"}
        for issue in review.issues
    ):
        review.passed = False

    review.missing_requirements = sorted(
        set(review.missing_requirements)
        | {
            issue.message
            for issue in review.issues
            if issue.category == "missing_requirement"
        }
    )
