import torch
import torch.nn as nn
import torch.nn.functional as F
from models import TransformerModel
from dataset import SequenceDataset

# Define the evaluation metrics

def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

            # Define the model evaluation function

    def evaluate_model(model, device, test_loader):
        evaluate(model, device, test_loader)

        # Define the main function

    def main():
        # Set the device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load the test data
        test_data = SequenceDataset('test_data.pth', 'test_labels.pth')
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=100, shuffle=False)

        # Load the model
        model = TransformerModel()
        model.to(device)

        # Evaluate the model
        evaluate_model(model, device, test_loader)

        if __name__ == '__main__':
            main()
