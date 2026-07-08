from ForgeCode.codeforge.generator import (
    normalize_python_indentation,
    normalize_python_string_literals,
    normalize_raw_artifact,
)
from ForgeCode.codeforge.schemas import (
    ImplementationPlan,
    PlannedFile,
)


def test_normalize_python_string_literals_escapes_raw_newline():
    broken = (
        "def evaluate():\n"
        "    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
        "'.format(loss, correct, total, acc))\n"
    )

    normalized = normalize_python_string_literals(broken)

    compile(normalized, "evaluate.py", "exec")
    assert "\\n'.format" in normalized


def test_normalize_raw_artifact_repairs_python_content_before_validation():
    plan = ImplementationPlan(
        project_name="demo",
        framework="pytorch",
        task_type="classification",
        summary="summary",
        dependencies=[],
        files=[
            PlannedFile(
                path="evaluate.py",
                purpose="evaluation",
                responsibilities=[],
                depends_on=[],
            )
        ],
        implementation_order=["evaluate.py"],
    )

    raw_artifact = {
        "path": "evaluate.py",
        "content": (
            "def evaluate():\n"
            "    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n"
            "'.format(loss, correct, total, acc))\n"
        ),
        "language": "python",
        "dependencies_used": [],
        "assumptions": [],
    }

    normalized = normalize_raw_artifact(
        raw_artifact=raw_artifact,
        target_path="evaluate.py",
        target_file=plan.files[0],
        dependency_context={},
        validated_plan=plan,
    )

    compile(normalized["content"], "evaluate.py", "exec")


def test_normalize_python_indentation_repairs_collapsed_blocks():
    broken = (
        "import torch\n"
        "import torch.nn as nn\n"
        "\n"
        "class PointDiT(nn.Module):\n"
        " def __init__(self):\n"
        " super(PointDiT, self).__init__()\n"
        " self.value = 1\n"
        " def forward(self, x):\n"
        " return x\n"
    )

    normalized = normalize_python_indentation(
        broken,
        path="models.py",
    )

    compile(normalized, "models.py", "exec")
    assert "    def __init__" in normalized
    assert "        super(" in normalized
