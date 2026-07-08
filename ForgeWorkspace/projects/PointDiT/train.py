import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np

# Define the training loop
def train(model, device, loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data, torch.randn(data.shape[0], 1).to(device), torch.randn(data.shape[0], 1).to(device))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Define the evaluation loop
def evaluate(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data, torch.randn(data.shape[0], 1).to(device), torch.randn(data.shape[0], 1).to(device))
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    accuracy = correct / len(loader.dataset)
    return test_loss / len(loader), accuracy

# Define the main function
def main():
    # Set the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = PointDiT()
    model.to(device)

    # Define the optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    # Load the dataset
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = PointDiTDataset('data', transform=transform)

    # Create the data loader
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    # Train the model
    for epoch in range(30):
        train(model, device, loader, optimizer, criterion, epoch)
        test_loss, accuracy = evaluate(model, device, loader)
        print(f'Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}')

if __name__ == '__main__':
    main()
