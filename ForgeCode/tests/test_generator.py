from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import asyncio
import json

from ForgeCode.codeforge.generator import generate_file


SPEC = """
Build a PyTorch 1D CNN classifier.

Input tensor shape:
(batch, 2, 1024)

Architecture:
- Conv1d: 2 input channels to 32 output channels,
  kernel size 7
- BatchNorm1d with 32 features
- ReLU
- MaxPool1d with kernel size 2
- Conv1d: 32 input channels to 64 output channels,
  kernel size 5
- BatchNorm1d with 64 features
- ReLU
- AdaptiveAvgPool1d output size 1
- Linear layer: 64 to 5 classes

Requirements:
- model implementation
- model configuration
- forward-pass shape test

Do not add a training pipeline.
Do not add dataset code.
"""


PLAN = {
    "project_name": "cnn_1d_classifier",
    "framework": "pytorch",
    "task_type": "1d_classification",
    "summary": (
        "Implement a compact 1D CNN classifier."
    ),
    "dependencies": [
        "torch"
    ],
    "files": [
        {
            "path": "configs/model.json",
            "purpose": (
                "Store model architecture parameters"
            ),
            "responsibilities": [
                "Define channel sizes",
                "Define kernel sizes",
                "Define number of classes"
            ],
            "depends_on": []
        },
        {
            "path": "src/models/model.py",
            "purpose": (
                "Implement the 1D CNN classifier"
            ),
            "responsibilities": [
                "Define convolution blocks",
                "Define pooling",
                "Define classifier",
                "Implement forward pass"
            ],
            "depends_on": [
                "configs/model.json"
            ]
        },
        {
            "path": "tests/test_model.py",
            "purpose": (
                "Verify model output shape"
            ),
            "responsibilities": [
                (
                    "Create input with shape "
                    "(batch, 2, 1024)"
                ),
                (
                    "Verify output shape is "
                    "(batch, 5)"
                )
            ],
            "depends_on": [
                "src/models/model.py"
            ]
        }
    ],
    "implementation_order": [
        "configs/model.json",
        "src/models/model.py",
        "tests/test_model.py"
    ],
    "assumptions": [],
    "unresolved_questions": []
}


async def main():

    result = await generate_file(
        spec=SPEC,
        plan=PLAN,
        target_path="configs/model.json"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    asyncio.run(
        main()
    )