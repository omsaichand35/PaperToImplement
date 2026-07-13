import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import PointDiT


class TestPointDiT(unittest.TestCase):
    def test_forward_pass(self):
        # Create a dummy input tensor
        x = torch.randn(1, 3, 256, 256)
        t = torch.randn(1, 256, 256, 3)

        # Initialize the model
        model = PointDiT()

        # Perform a forward pass
        output = model(x, t)

        # Check the output shape
        self.assertEqual(output.shape, (1, 256, 256, 3))

if __name__ == '__main__':
    unittest.main()
