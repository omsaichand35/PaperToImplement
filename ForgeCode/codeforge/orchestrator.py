from copy import deepcopy

from .generator import generate_file
from .implementor import implement_review
from .planner import create_implementation_plan
from .reviewer import review_implementation

from .schemas import (
    GeneratedFile,
    ImplementationPlan,
    ImplementationReview,
)


async def forge_implementation(
    spec: str,
    max_review_rounds: int = 3,
) -> dict:
    """
    Run the complete ForgeCode pipeline:

    spec
      -> plan
      -> generate artifacts
      -> review
      -> implement fixes
      -> re-review

    Returns the final plan, artifacts,
    review history, and status.
    """

    spec = spec.strip()

    if not spec:
        raise ValueError(
            "spec cannot be empty"
        )

    if max_review_rounds < 1:
        raise ValueError(
            "max_review_rounds must be at least 1"
        )

    # ---------------------------------
    # 1. Create and validate plan
    # ---------------------------------
    print("[ForgeCode] Step 1: Creating implementation plan...", flush=True)
    raw_plan = await create_implementation_plan(
        spec=spec
    )

    plan = ImplementationPlan.model_validate(
        raw_plan
    )
    print(f"[ForgeCode] Plan created for project '{plan.project_name}' ({len(plan.implementation_order)} files planned).", flush=True)

    # ---------------------------------
    # 2. Generate all planned files
    # ---------------------------------
    print(f"[ForgeCode] Step 2: Generating {len(plan.implementation_order)} files in dependency order...", flush=True)
    artifacts_by_path: dict[
        str,
        GeneratedFile
    ] = {}

    for idx, target_path in enumerate(plan.implementation_order, 1):
        print(f"[ForgeCode]   -> ({idx}/{len(plan.implementation_order)}) Generating {target_path}...", flush=True)
        dependency_context = (
            build_dependency_context(
                plan=plan,
                target_path=target_path,
                artifacts_by_path=artifacts_by_path,
            )
        )

        error_context = None
        for attempt in range(1, 5):
            try:
                raw_artifact = await generate_file(
                    spec=spec,
                    plan=plan.model_dump(),
                    target_path=target_path,
                    dependency_context=dependency_context,
                    error_context=error_context
                )

                artifact = GeneratedFile.model_validate(
                    raw_artifact
                )

                artifacts_by_path[
                    artifact.path
                ] = artifact
                break
            except Exception as exc:
                if attempt == 4:
                    raise
                print(f"[ForgeCode]      -> Attempt {attempt} failed validation: {exc}. Retrying with self-correction feedback...", flush=True)
                error_context = str(exc)

    validate_complete_artifact_set(
        plan=plan,
        artifacts_by_path=artifacts_by_path,
    )
    print("[ForgeCode] All planned files generated and validated.", flush=True)

    # ---------------------------------
    # 3. Review and repair loop
    # ---------------------------------
    print(f"[ForgeCode] Step 3: Starting review & repair loop (max rounds: {max_review_rounds})...", flush=True)
    review_history: list[
        ImplementationReview
    ] = []

    for round_number in range(
        1,
        max_review_rounds + 1
    ):
        print(f"[ForgeCode]   -> Review Round {round_number}/{max_review_rounds}: Inspecting {len(artifacts_by_path)} files...", flush=True)
        current_artifacts = (
            ordered_artifacts(
                plan=plan,
                artifacts_by_path=artifacts_by_path,
            )
        )

        raw_review = await review_implementation(
            spec=spec,
            plan=plan.model_dump(),
            artifacts=[
                artifact.model_dump()
                for artifact
                in current_artifacts
            ],
        )

        review = (
            ImplementationReview.model_validate(
                raw_review
            )
        )

        review_history.append(
            review
        )

        # Review passed.
        if review.passed:
            print(f"[ForgeCode]   -> Review Round {round_number}: PASSED! All checks clean.", flush=True)
            return build_result(
                plan=plan,
                artifacts_by_path=artifacts_by_path,
                review_history=review_history,
                status="passed",
                review_rounds=round_number,
            )

        # No repair round remains.
        if round_number >= max_review_rounds:
            print(f"[ForgeCode]   -> Review Round {round_number}: FAILED (exhausted max rounds).", flush=True)
            break

        error_context = None
        raw_fixed_artifacts = []
        for attempt in range(1, 4):
            try:
                raw_fixed_artifacts = await implement_review(
                    spec=spec,
                    plan=plan.model_dump(),
                    review=review.model_dump(),
                    artifacts=[
                        artifact.model_dump()
                        for artifact
                        in current_artifacts
                    ],
                    error_context=error_context
                )
                
                # Check validation manually inside the try block to catch validation errors
                fixed_artifacts = [
                    GeneratedFile.model_validate(
                        artifact
                    )
                    for artifact
                    in raw_fixed_artifacts
                ]
                break
            except Exception as exc:
                if attempt == 3:
                    raise
                print(f"[ForgeCode]      -> Repair Attempt {attempt} failed validation: {exc}. Retrying with self-correction feedback...", flush=True)
                error_context = str(exc)

        if not fixed_artifacts:
            raise RuntimeError(
                "Review failed but implementor "
                "returned no fixed artifacts"
            )

        merge_fixed_artifacts(
            artifacts_by_path=artifacts_by_path,
            fixed_artifacts=fixed_artifacts,
        )

        validate_complete_artifact_set(
            plan=plan,
            artifacts_by_path=artifacts_by_path,
        )

    # ---------------------------------
    # 4. Review budget exhausted
    # ---------------------------------

    return build_result(
        plan=plan,
        artifacts_by_path=artifacts_by_path,
        review_history=review_history,
        status="review_failed",
        review_rounds=len(review_history),
    )


def build_dependency_context(
    plan: ImplementationPlan,
    target_path: str,
    artifacts_by_path: dict[
        str,
        GeneratedFile
    ],
) -> dict[str, str]:
    """
    Build dependency context only from
    already-generated declared dependencies.
    """

    target_file = next(
        (
            file
            for file in plan.files
            if file.path == target_path
        ),
        None,
    )

    if target_file is None:
        raise ValueError(
            f"Target path is not planned: "
            f"{target_path}"
        )

    context: dict[str, str] = {}

    for dependency_path in target_file.depends_on:

        dependency_artifact = (
            artifacts_by_path.get(
                dependency_path
            )
        )

        if dependency_artifact is None:
            raise RuntimeError(
                f"Dependency {dependency_path} "
                f"required by {target_path} "
                "has not been generated yet"
            )

        context[
            dependency_path
        ] = dependency_artifact.content

    return context


def ordered_artifacts(
    plan: ImplementationPlan,
    artifacts_by_path: dict[
        str,
        GeneratedFile
    ],
) -> list[GeneratedFile]:
    """
    Return artifacts in validated
    implementation order.
    """

    return [
        artifacts_by_path[path]
        for path in plan.implementation_order
    ]


def validate_complete_artifact_set(
    plan: ImplementationPlan,
    artifacts_by_path: dict[
        str,
        GeneratedFile
    ],
) -> None:
    """
    Ensure the generated artifact set exactly
    matches the planned file set.
    """

    planned_paths = {
        file.path
        for file in plan.files
    }

    artifact_paths = set(
        artifacts_by_path.keys()
    )

    missing_paths = (
        planned_paths
        - artifact_paths
    )

    if missing_paths:
        raise RuntimeError(
            "Missing generated artifacts: "
            f"{sorted(missing_paths)}"
        )

    unexpected_paths = (
        artifact_paths
        - planned_paths
    )

    if unexpected_paths:
        raise RuntimeError(
            "Unexpected generated artifacts: "
            f"{sorted(unexpected_paths)}"
        )


def merge_fixed_artifacts(
    artifacts_by_path: dict[
        str,
        GeneratedFile
    ],
    fixed_artifacts: list[
        GeneratedFile
    ],
) -> None:
    """
    Replace only existing artifacts with
    validated repaired versions.
    """

    for artifact in fixed_artifacts:

        if artifact.path not in artifacts_by_path:
            raise RuntimeError(
                "Cannot merge unknown fixed "
                f"artifact: {artifact.path}"
            )

        artifacts_by_path[
            artifact.path
        ] = artifact


def build_result(
    plan: ImplementationPlan,
    artifacts_by_path: dict[
        str,
        GeneratedFile
    ],
    review_history: list[
        ImplementationReview
    ],
    status: str,
    review_rounds: int,
) -> dict:
    """
    Build the final orchestration result.
    """

    artifacts = ordered_artifacts(
        plan=plan,
        artifacts_by_path=artifacts_by_path,
    )

    return {
        "status": status,
        "review_rounds": review_rounds,
        "plan": plan.model_dump(),
        "artifacts": [
            artifact.model_dump()
            for artifact in artifacts
        ],
        "final_review": (
            review_history[-1].model_dump()
            if review_history
            else None
        ),
        "review_history": [
            review.model_dump()
            for review in review_history
        ],
    }
