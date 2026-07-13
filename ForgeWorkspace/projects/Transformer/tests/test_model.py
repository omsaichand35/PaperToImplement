import pytest
import torch
import torch.nn as nn
from models import TransformerModel

# Test the forward pass of the Transformer model

def test_transformer_model_forward_pass():
    # Create a dummy input sequence
    input_seq = torch.randn(1, 10, 512)

    # Create an instance of the Transformer model
    model = TransformerModel()

    # Perform the forward pass
    output = model(input_seq)

    # Check the output shape
    assert output.shape == (1, 10, 512)
