import re
from dataclasses import dataclass

from .schemas import (
    GeneratedFile,
    ReviewIssue,
)


@dataclass(frozen=True)
class ContractRule:
    name: str
    description: str
    affected_files: tuple[str, ...]
    required_patterns: tuple[str, ...]
    evidence_terms: tuple[str, ...]
    forbidden_patterns: tuple[str, ...] = ()
    recommendation: str | None = None


def build_contract_rules(
    spec: str,
    artifacts: list[GeneratedFile],
) -> list[ContractRule]:
    """
    Derive architecture-level invariants from the paper spec.

    These checks intentionally stay broad and domain-shaped:
    they prevent common template drift without trying to fully
    prove semantic correctness.
    """

    text = spec.lower()
    artifact_paths = {
        artifact.path
        for artifact in artifacts
    }

    def existing(paths: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(
            path
            for path in paths
            if path in artifact_paths
        )

    rules: list[ContractRule] = []

    if "dino" in text:
        rules.append(
            ContractRule(
                name="missing_dino_conditioning",
                description=(
                    "Specification mentions DINO/DINOv3 conditioning, "
                    "but the model code does not implement a DINO feature branch."
                ),
                affected_files=existing(("models.py",)),
                required_patterns=(r"\bdino\b",),
                evidence_terms=("DINO", "DINOv3"),
                recommendation=(
                    "Add a frozen DINO feature extractor or explicitly record "
                    "DINO as unresolved if unavailable."
                ),
            )
        )

    if "patchify" in text or "patchification" in text or "unpatchify" in text:
        rules.append(
            ContractRule(
                name="missing_patch_token_pipeline",
                description=(
                    "Specification describes patchify/unpatchify token flow, "
                    "but the architecture lacks that pipeline."
                ),
                affected_files=existing(("models.py",)),
                required_patterns=(r"patchify|patch_embed|unfold", r"unpatchify|fold"),
                evidence_terms=("patchify", "unpatchify", "patchification"),
                recommendation=(
                    "Implement explicit patch embedding and reconstruction "
                    "or mark the exact patch pipeline as unresolved."
                ),
            )
        )

    if "diffusion" in text or "flow matching" in text:
        rules.append(
            ContractRule(
                name="missing_diffusion_time_interface",
                description=(
                    "Specification describes diffusion or flow matching, "
                    "but the model code lacks timestep/noise conditioning."
                ),
                affected_files=existing(("models.py", "train.py")),
                required_patterns=(r"\btimestep\b|\btime_embed\b|\btime_mlp\b|\bt\s*[:,)]", r"\bnoise\b|\bepsilon\b|\bzt\b|\bz_t\b"),
                evidence_terms=("diffusion", "flow matching", "time step", "noise"),
                recommendation=(
                    "Expose timestep and noisy-sample inputs in the model "
                    "and train with the paper's diffusion/flow objective."
                ),
            )
        )

    if "flow matching" in text or "velocity loss" in text or "x-prediction" in text:
        rules.append(
            ContractRule(
                name="missing_flow_matching_objective",
                description=(
                    "Specification describes flow matching/x-prediction, "
                    "but training does not implement that objective."
                ),
                affected_files=existing(("train.py",)),
                required_patterns=(r"flow|velocity|v_loss|x_prediction|x_pred", r"logit|sigmoid|p_zero|pzero|t\s*="),
                evidence_terms=("flow matching", "x-prediction", "velocity loss"),
                recommendation=(
                    "Implement timestep sampling, noisy interpolation, "
                    "clean prediction, and velocity-space loss."
                ),
            )
        )

    if "point map normalization" in text or (
        "centroid" in text and "scale" in text and "point map" in text
    ):
        rules.append(
            ContractRule(
                name="missing_point_map_normalization",
                description=(
                    "Specification requires point map normalization, "
                    "but preprocessing does not compute centroid/scale normalization."
                ),
                affected_files=existing(("dataset.py", "utils.py", "train.py")),
                required_patterns=(r"centroid|mean", r"scale|norm|euclidean"),
                evidence_terms=("point map normalization", "centroid", "scale"),
                recommendation=(
                    "Normalize point maps per sample and preserve metadata "
                    "needed for denormalization or affine-invariant evaluation."
                ),
            )
        )

    if any(term in text for term in ("relp", "reld", "bf1", "affine-invariant", "delta", "δ")):
        rules.append(
            ContractRule(
                name="missing_dense_geometry_metrics",
                description=(
                    "Specification describes dense geometry metrics, "
                    "but evaluation appears to use generic classification metrics."
                ),
                affected_files=existing(("evaluate.py",)),
                required_patterns=(r"rel|delta|bf1|affine|scale.*shift"),
                forbidden_patterns=(r"torch\.max\s*\(", r"accuracy|correct\s*\+="),
                evidence_terms=("Rel", "delta", "BF1", "affine-invariant"),
                recommendation=(
                    "Implement the paper's dense prediction metrics instead "
                    "of classification accuracy."
                ),
            )
        )

    return rules


def contract_issues(
    spec: str,
    artifacts: list[GeneratedFile],
) -> list[ReviewIssue]:
    issues: list[ReviewIssue] = []
    content_by_path = {
        artifact.path: artifact.content
        for artifact in artifacts
    }

    for rule in build_contract_rules(spec, artifacts):
        if not rule.affected_files:
            continue

        scoped_content = "\n".join(
            content_by_path.get(path, "")
            for path in rule.affected_files
        )

        missing_required = [
            pattern
            for pattern in rule.required_patterns
            if not pattern_matches(pattern, scoped_content)
        ]
        present_forbidden = [
            pattern
            for pattern in rule.forbidden_patterns
            if pattern_matches(pattern, scoped_content)
        ]

        if not missing_required and not present_forbidden:
            continue

        evidence = [
            f"Spec contains: {term}"
            for term in rule.evidence_terms
            if term.lower() in spec.lower()
        ]
        if missing_required:
            evidence.append(
                "Missing implementation patterns: "
                + ", ".join(missing_required)
            )
        if present_forbidden:
            evidence.append(
                "Found incompatible implementation patterns: "
                + ", ".join(present_forbidden)
            )

        issues.append(
            ReviewIssue(
                severity="critical",
                category="missing_requirement",
                message=rule.description,
                affected_files=list(rule.affected_files),
                evidence=evidence,
                recommendation=rule.recommendation,
            )
        )

    return issues


def pattern_matches(
    pattern: str,
    text: str,
) -> bool:
    try:
        return re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        ) is not None
    except re.PatternError:
        return pattern.lower() in text.lower()
