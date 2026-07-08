import pytest
import torch
import torch.nn as nn
from models import PointDiT

# Test the forward pass of the model
def test_forward_pass():
    model = PointDiT()
    dummy_input = torch.randn(1, 3, 512, 512)
    output = model(dummy_input)
    assert output.shape == (1, 3*16*16)

# Test the model's output shape\ndef test_output_shape():\n    model = PointDiT()\n    dummy_input = torch.randn(1, 3, 512, 512)\n    output = model(dummy_input)\n    assert output.shape == (1, 3*16*16)\n