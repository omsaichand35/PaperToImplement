from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import asyncio
import json

from ForgeCode.codeforge.reviewer import (
    review_implementation
)


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
    "summary": "Implement a compact 1D CNN classifier.",
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
                "Create input with shape (batch, 2, 1024)",
                "Verify output shape is (batch, 5)"
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


ARTIFACTS = [
    {
        "path": "configs/model.json",
        "content": json.dumps({
            "in_channels": 2,
            "out_channels_1": 32,
            "kernel_size_1": 7,
            "out_channels_2": 64,
            "kernel_size_2": 5,
            "num_classes": 5
        }, indent=2),
        "language": "json",
        "dependencies_used": [],
        "assumptions": []
    },
    {
        "path": "src/models/model.py",
        "content": (
            "import json\n"
            "import torch.nn as nn\n"
            "\n"
            "\n"
            "class CNN1DClassifier(nn.Module):\n"
            "\n"
            "    def __init__(self, config_path):\n"
            "        super().__init__()\n"
            "        with open(config_path) as f:\n"
            "            cfg = json.load(f)\n"
            "\n"
            "        self.conv1 = nn.Conv1d(\n"
            "            cfg['in_channels'],\n"
            "            cfg['out_channels_1'],\n"
            "            cfg['kernel_size_1']\n"
            "        )\n"
            "        self.bn1 = nn.BatchNorm1d(\n"
            "            cfg['out_channels_1']\n"
            "        )\n"
            "        self.relu = nn.ReLU()\n"
            "        self.pool = nn.MaxPool1d(2)\n"
            "\n"
            "        self.conv2 = nn.Conv1d(\n"
            "            cfg['out_channels_1'],\n"
            "            cfg['out_channels_2'],\n"
            "            cfg['kernel_size_2']\n"
            "        )\n"
            "        self.bn2 = nn.BatchNorm1d(\n"
            "            cfg['out_channels_2']\n"
            "        )\n"
            "        self.adaptive_pool = (\n"
            "            nn.AdaptiveAvgPool1d(1)\n"
            "        )\n"
            "        self.fc = nn.Linear(\n"
            "            cfg['out_channels_2'],\n"
            "            cfg['num_classes']\n"
            "        )\n"
            "\n"
            "    def forward(self, x):\n"
            "        x = self.pool(\n"
            "            self.relu(\n"
            "                self.bn1(self.conv1(x))\n"
            "            )\n"
            "        )\n"
            "        x = self.relu(\n"
            "            self.bn2(self.conv2(x))\n"
            "        )\n"
            "        x = self.adaptive_pool(x)\n"
            "        x = x.squeeze(-1)\n"
            "        return self.fc(x)\n"
        ),
        "language": "python",
        "dependencies_used": [
            "configs/model.json"
        ],
        "assumptions": []
    },
    {
        "path": "tests/test_model.py",
        "content": (
            "import torch\n"
            "from src.models.model import CNN1DClassifier\n"
            "\n"
            "\n"
            "def test_output_shape():\n"
            "    model = CNN1DClassifier(\n"
            "        'configs/model.json'\n"
            "    )\n"
            "    x = torch.randn(4, 2, 1024)\n"
            "    out = model(x)\n"
            "    assert out.shape == (4, 5)\n"
        ),
        "language": "python",
        "dependencies_used": [
            "src/models/model.py"
        ],
        "assumptions": []
    }
]


async def main():

    result = await review_implementation(
        spec=SPEC,
        plan=PLAN,
        artifacts=ARTIFACTS
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
