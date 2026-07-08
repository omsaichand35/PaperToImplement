import torch
import torch.nn as nn
import torchvision
from models import PointDiT
from dataset import PointDiTDataset

# Define the evaluation metrics
def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, torch.randn(data.shape[0], 1).to(device), torch.randn(data.shape[0], 1).to(device))
            test_loss += nn.CrossEntropyLoss()(output, target).item()
            _, predicted = torch.max(output, 1)
            correct += (predicted == target).sum().item()

    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Define the main function
def main():
    # Set the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = PointDiT()
    model.to(device)

    # Load the test dataset
    test_dataset = PointDiTDataset('data', transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluate the model
    evaluate(model, device, test_loader)

if __name__ == '__main__':
    main()
