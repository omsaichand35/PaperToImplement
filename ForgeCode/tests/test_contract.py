from ForgeCode.codeforge.contract import contract_issues
from ForgeCode.codeforge.reviewer import apply_contract_issues
from ForgeCode.codeforge.schemas import (
    GeneratedFile,
    ImplementationReview,
)


POINTDIT_SPEC = """
## Architecture Contract
- If DINO/DINOv3 is specified, models.py must include a DINO feature-extractor branch.
- If diffusion or flow matching is specified, the model interface must include noisy sample and timestep inputs.
- If patchify/unpatchify is specified, models.py must implement explicit patch/token embedding and reconstruction.
- If point-map normalization is specified, dataset.py or utils.py must compute centroid/scale normalization.
- If dense geometry metrics are specified, evaluate.py must implement Relp, Reld, δ1, BF1, and affine-invariant alignment.

The paper uses a pixel-space diffusion transformer with flow matching,
x-prediction, DINOv3 image conditioning, patchify/unpatchify, point map
normalization by centroid and scale, and dense geometry metrics Relp,
Reld, δ1, BF1.
"""


def test_contract_issues_catch_pointdit_template_drift():
    artifacts = [
        GeneratedFile(
            path="models.py",
            content=(
                "import torchvision\n"
                "import torch.nn as nn\n"
                "class Model(nn.Module):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.vit = torchvision.models.vit_b_16(pretrained=True)\n"
                "        self.head = nn.Conv2d(3, 3, 3)\n"
                "    def forward(self, x):\n"
                "        return self.head(self.vit(x))\n"
            ),
            language="python",
            dependencies_used=["torchvision"],
            assumptions=[],
        ),
        GeneratedFile(
            path="dataset.py",
            content="def load_point_map(path):\n    return np.load(path)\n",
            language="python",
            dependencies_used=[],
            assumptions=[],
        ),
        GeneratedFile(
            path="train.py",
            content="loss_fn = nn.MSELoss()\n",
            language="python",
            dependencies_used=[],
            assumptions=[],
        ),
        GeneratedFile(
            path="evaluate.py",
            content=(
                "_, predicted = torch.max(output, 1)\n"
                "correct += (predicted == target).sum().item()\n"
            ),
            language="python",
            dependencies_used=[],
            assumptions=[],
        ),
    ]

    issue_names = {
        issue.message
        for issue in contract_issues(
            spec=POINTDIT_SPEC,
            artifacts=artifacts,
        )
    }

    assert any("DINO" in message for message in issue_names)
    assert any("timestep/noise" in message for message in issue_names)
    assert any("patchify/unpatchify" in message for message in issue_names)
    assert any("point map normalization" in message for message in issue_names)
    assert any("dense geometry metrics" in message for message in issue_names)


def test_apply_contract_issues_turns_passed_review_into_failure():
    review = ImplementationReview(
        passed=True,
        summary="LLM reviewer missed the issue",
        issues=[],
        checked_files=["models.py"],
        missing_requirements=[],
        invented_details=[],
        cross_file_inconsistencies=[],
    )
    artifacts = [
        GeneratedFile(
            path="models.py",
            content="class Model:\n    def forward(self, x):\n        return x\n",
            language="python",
            dependencies_used=[],
            assumptions=[],
        )
    ]

    apply_contract_issues(
        review=review,
        spec="The paper uses DINOv3 image conditioning.",
        artifacts=artifacts,
    )

    assert review.passed is False
    assert review.issues
    assert review.issues[0].affected_files == ["models.py"]
