from ForgeCode.codeforge.implementor import (
    normalize_fixed_artifacts,
    select_affected_artifacts,
    validate_fixed_content_format,
)
from ForgeCode.codeforge.schemas import GeneratedFile


def test_select_affected_artifacts_limits_repair_scope():
    artifacts = [
        GeneratedFile(
            path="models.py",
            content="class A:\n    pass\n",
            language="python",
            dependencies_used=[],
            assumptions=[],
        ),
        GeneratedFile(
            path="train.py",
            content="def train():\n    pass\n",
            language="python",
            dependencies_used=[],
            assumptions=[],
        ),
    ]

    affected = select_affected_artifacts(
        artifacts=artifacts,
        affected_paths={"models.py"},
    )

    assert [artifact.path for artifact in affected] == ["models.py"]


def test_normalize_fixed_artifacts_repairs_python_string_newlines():
    fixed = normalize_fixed_artifacts([
        {
            "path": "evaluate.py",
            "content": (
                "def evaluate():\n"
                "    print('broken string\n"
                "'.format())\n"
            ),
            "language": "python",
            "dependencies_used": [],
            "assumptions": [],
        }
    ])

    compile(fixed[0]["content"], "evaluate.py", "exec")


def test_validate_fixed_content_format_reports_snippet():
    artifacts = [
        GeneratedFile(
            path="models.py",
            content="def broken():\nprint('oops')\n",
            language="python",
            dependencies_used=[],
            assumptions=[],
        )
    ]

    try:
        validate_fixed_content_format(artifacts)
    except ValueError as error:
        message = str(error)
        assert "models.py on line 2" in message
        assert "Code snippet around line 2" in message
    else:
        raise AssertionError("Expected syntax validation to fail")
