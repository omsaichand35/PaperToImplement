import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # [Paper Spec Section 3.3] Convolutional Layer 1 (Convolutional)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2)
        # [Paper Spec Section 3.3] ReLU Nonlinearity (Activation)
        self.relu1 = nn.ReLU(inplace=True)
        # [Paper Spec Section 3.3] Local Response Normalization (Normalization)
        self.norm1 = nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=2)
        # [Paper Spec Section 3.3] Max Pooling (Pooling)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        # [Paper Spec Section 3.4] Convolutional Layer 2 (Convolutional)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        # [Paper Spec Section 3.4] ReLU Nonlinearity (Activation)
        self.relu2 = nn.ReLU(inplace=True)
        # [Paper Spec Section 3.4] Local Response Normalization (Normalization)
        self.norm2 = nn.LocalResponseNorm(2, alpha=1e-4, beta=0.75, k=2)
        # [Paper Spec Section 3.4] Max Pooling (Pooling)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)

        # [Paper Spec Section 3.5] Convolutional Layer 3 (Convolutional)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        # [Paper Spec Section 3.5] ReLU Nonlinearity (Activation)
        self.relu3 = nn.ReLU(inplace=True)

        # [Paper Spec Section 3.5] Convolutional Layer 4 (Convolutional)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        # [Paper Spec Section 3.5] ReLU Nonlinearity (Activation)
        self.relu4 = nn.ReLU(inplace=True)

        # [Paper Spec Section 3.5] Convolutional Layer 5 (Convolutional)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        # [Paper Spec Section 3.5] ReLU Nonlinearity (Activation)
        self.relu5 = nn.ReLU(inplace=True)
        # [Paper Spec Section 3.5] Max Pooling (Pooling)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)

        # [Paper Spec Section 3.6] Fully Connected Layer 1 (Fully Connected)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        # [Paper Spec Section 3.6] ReLU Nonlinearity (Activation)
        self.relu6 = nn.ReLU(inplace=True)
        # [Paper Spec Section 3.6] Dropout
        self.drop6 = nn.Dropout(p=0.5)

        # [Paper Spec Section 3.6] Fully Connected Layer 2 (Fully Connected)
        self.fc7 = nn.Linear(4096, 4096)
        # [Paper Spec Section 3.6] ReLU Nonlinearity (Activation)
        self.relu7 = nn.ReLU(inplace=True)
        # [Paper Spec Section 3.6] Dropout
        self.drop7 = nn.Dropout(p=0.5)

        # [Paper Spec Section 3.6] Fully Connected Layer 3 (Fully Connected)
        self.fc8 = nn.Linear(4096, 1000)

    def forward(self, x):
        # [Paper Spec Section 3.3] Convolutional Layer 1 (Convolutional)
        x = self.conv1(x)
        # [Paper Spec Section 3.3] ReLU Nonlinearity (Activation)
        x = self.relu1(x)
        # [Paper Spec Section 3.3] Local Response Normalization (Normalization)
        x = self.norm1(x)
        # [Paper Spec Section 3.3] Max Pooling (Pooling)
        x = self.pool1(x)

        # [Paper Spec Section 3.4] Convolutional Layer 2 (Convolutional)
        x = self.conv2(x)
        # [Paper Spec Section 3.4] ReLU Nonlinearity (Activation)
        x = self.relu2(x)
        # [Paper Spec Section 3.4] Local Response Normalization (Normalization)
        x = self.norm2(x)
        # [Paper Spec Section 3.4] Max Pooling (Pooling)
        x = self.pool2(x)

        # [Paper Spec Section 3.5] Convolutional Layer 3 (Convolutional)
        x = self.conv3(x)
        # [Paper Spec Section 3.5] ReLU Nonlinearity (Activation)
        x = self.relu3(x)

        # [Paper Spec Section 3.5] Convolutional Layer 4 (Convolutional)
        x = self.conv4(x)
        # [Paper Spec Section 3.5] ReLU Nonlinearity (Activation)
        x = self.relu4(x)

        # [Paper Spec Section 3.5] Convolutional Layer 5 (Convolutional)
        x = self.conv5(x)
        # [Paper Spec Section 3.5] ReLU Nonlinearity (Activation)
        x = self.relu5(x)
        # [Paper Spec Section 3.5] Max Pooling (Pooling)
        x = self.pool5(x)

        # [Paper Spec Section 3.6] Flatten
        x = x.view(-1, 256 * 6 * 6)

        # [Paper Spec Section 3.6] Fully Connected Layer 1 (Fully Connected)
        x = self.fc6(x)
        # [Paper Spec Section 3.6] ReLU Nonlinearity (Activation)
        x = self.relu6(x)
        # [Paper Spec Section 3.6] Dropout
        x = self.drop6(x)

        # [Paper Spec Section 3.6] Fully Connected Layer 2 (Fully Connected)
        x = self.fc7(x)
        # [Paper Spec Section 3.6] ReLU Nonlinearity (Activation)
        x = self.relu7(x)
        # [Paper Spec Section 3.6] Dropout
        x = self.drop7(x)

        # [Paper Spec Section 3.6] Fully Connected Layer 3 (Fully Connected)
        x = self.fc8(x)

        return x

# Initialize the model
model = AlexNet()
