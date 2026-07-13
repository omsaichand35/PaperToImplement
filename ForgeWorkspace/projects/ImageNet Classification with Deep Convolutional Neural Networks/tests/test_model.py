import pytest
import torch
import torch.nn as nn
from models import AlexNet


def test_forward_pass():
    # Initialize the model
    model = AlexNet(num_classes=1000)

    # Create a dummy input
    input_tensor = torch.randn(1, 3, 224, 224)

    # Perform a forward pass
    output = model(input_tensor)

    # Check the output shape
    assert output.shape == (1, 1000)
