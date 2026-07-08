from ForgeCode.codeforge.generator import validate_generated_artifact
from ForgeCode.codeforge.reviewer import validate_review_evidence
from ForgeCode.codeforge.schemas import GeneratedFile, ImplementationPlan, ImplementationReview, ReviewIssue, PlannedFile

def test_dependency_normalization():
    # Setup test variables
    planned_paths = {"config.py", "src/models/model.py", "tests/test_model.py"}
    
    # 1. Dependency with extension omitted (e.g. 'config')
    artifact = GeneratedFile(
        path="src/models/model.py",
        content="import config",
        language="python",
        dependencies_used=["config"],
        assumptions=[]
    )
    
    # Should not raise any exceptions and should normalize 'config' to 'config.py'
    validate_generated_artifact(artifact, "src/models/model.py", planned_paths)
    assert artifact.dependencies_used == ["config.py"]

    # 2. Dependency in subdirectory with path/extension omitted (e.g. 'model')
    artifact_test = GeneratedFile(
        path="tests/test_model.py",
        content="import model",
        language="python",
        dependencies_used=["model"],
        assumptions=[]
    )
    validate_generated_artifact(artifact_test, "tests/test_model.py", planned_paths)
    assert artifact_test.dependencies_used == ["src/models/model.py"]

    # 3. Submodule of allowed external packages (e.g., 'torch.nn')
    artifact_torch = GeneratedFile(
        path="src/models/model.py",
        content="import torch.nn",
        language="python",
        dependencies_used=["torch.nn"],
        assumptions=[]
    )
    validate_generated_artifact(artifact_torch, "src/models/model.py", planned_paths)
    assert artifact_torch.dependencies_used == ["torch.nn"]


def test_reviewer_path_normalization():
    plan = ImplementationPlan(
        project_name="test_proj",
        framework="pytorch",
        task_type="classification",
        summary="summary",
        dependencies=[],
        files=[
            PlannedFile(path="config.py", purpose="config", responsibilities=[]),
            PlannedFile(path="src/models/model.py", purpose="model", responsibilities=[])
        ],
        implementation_order=["config.py", "src/models/model.py"]
    )
    
    artifacts = [
        GeneratedFile(path="config.py", content="", language="python", dependencies_used=[]),
        GeneratedFile(path="src/models/model.py", content="", language="python", dependencies_used=["config.py"])
    ]
    
    # Review with non-fully-qualified paths and single string for 'evidence' list field
    review = ImplementationReview(
        passed=False,
        checked_files=["config", "model"],
        issues=[
            ReviewIssue(
                severity="error",
                category="dependency_mismatch",
                message="test issue",
                affected_files=["model"],
                evidence="single string evidence",
                recommendation="fix it"
            )
        ],
        summary="summary"
    )
    
    validate_review_evidence(review, plan, artifacts)
    
    # Assert they have been normalized back to full paths
    assert review.checked_files == ["config.py", "src/models/model.py"]
    assert review.issues[0].affected_files == ["src/models/model.py"]
    # Assert evidence was coerced to a list
    assert review.issues[0].evidence == ["single string evidence"]


def test_reviewer_does_not_broaden_ambiguous_issue_paths():
    plan = ImplementationPlan(
        project_name="test_proj",
        framework="pytorch",
        task_type="classification",
        summary="summary",
        dependencies=[],
        files=[
            PlannedFile(path="config.py", purpose="config", responsibilities=[]),
            PlannedFile(path="src/models/model.py", purpose="model", responsibilities=[]),
        ],
        implementation_order=["config.py", "src/models/model.py"],
    )

    artifacts = [
        GeneratedFile(path="config.py", content="", language="python", dependencies_used=[]),
        GeneratedFile(path="src/models/model.py", content="", language="python", dependencies_used=["config.py"]),
    ]

    review = ImplementationReview(
        passed=False,
        checked_files=["config", "model"],
        issues=[
            ReviewIssue(
                severity="error",
                category="other",
                message="ambiguous issue path",
                affected_files=["implementation_plan"],
                evidence=["meta reference"],
                recommendation="fix it",
            )
        ],
        summary="summary",
    )

    validate_review_evidence(review, plan, artifacts)

    assert review.issues[0].affected_files == []

if __name__ == "__main__":
    test_dependency_normalization()
    test_reviewer_path_normalization()
    test_reviewer_does_not_broaden_ambiguous_issue_paths()
    print("All normalization tests passed successfully!")
