from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ForgeCore"))
sys.path.insert(0, str(ROOT / "ForgeCode"))

from core import _build_runtime_repair_review


def test_runtime_repair_review_extracts_attribute_and_name_errors():
    artifacts_by_path = {
        "models.py": {
            "path": "models.py",
            "content": (
                "class PointDiT:\n"
                "    def __init__(self):\n"
                "        self.dino_encoder = torchvision.models.dino.dino_vits16()\n"
                "\n"
                "def patchify(x):\n"
                "    return x\n"
            ),
            "language": "python",
            "dependencies_used": [],
            "assumptions": [],
        },
        "tests/test_model.py": {
            "path": "tests/test_model.py",
            "content": "from models import PointDiT\nunpatchify(None, 16)\n",
            "language": "python",
            "dependencies_used": [],
            "assumptions": [],
        },
    }
    error_output = """
models.py:12: AttributeError: module 'torchvision.models' has no attribute 'dino'
tests/test_model.py:24: NameError: name 'unpatchify' is not defined
"""

    review = _build_runtime_repair_review(
        failed_stage="pytest",
        error_output=error_output,
        enriched_error=error_output,
        artifacts_by_path=artifacts_by_path,
    )

    messages = [issue["message"] for issue in review["issues"]]
    assert any("does not provide attribute 'dino'" in msg for msg in messages)
    assert any("symbol 'unpatchify'" in msg for msg in messages)
    assert "models.py" in review["_affected_paths"]
    assert "tests/test_model.py" in review["_affected_paths"]
