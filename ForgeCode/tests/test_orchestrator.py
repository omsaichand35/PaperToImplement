from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import asyncio
import json

from ForgeCode.codeforge.orchestrator import (
    forge_implementation
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


async def main():

    result = await forge_implementation(
        spec=SPEC,
        max_review_rounds=3
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
