import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import PointDiT
from dataset import train_dataset

# Define hyperparameters
batch_size = 32
epochs = 30
learning_rate = 5e-5

# Create data loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
model = PointDiT()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0)
loss_fn = nn.MSELoss()

# Train the model
for epoch in range(epochs):
    for batch in train_loader:
        # Get input and target tensors
        input_tensor, target_tensor = batch

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(input_tensor, torch.zeros((input_tensor.shape[0],)))

        # Calculate loss
        loss = loss_fn(output, target_tensor)

        # Backward pass
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Print loss at each epoch
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
