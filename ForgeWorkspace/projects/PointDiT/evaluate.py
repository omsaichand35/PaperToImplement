import torch
import torch.nn as nn
import torchvision
from models import get_model
from dataset import get_dataset

# Evaluation metrics

def accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / len(target)
    return accuracy

    # Evaluation loop

    def evaluate(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, torch.randn(data.shape[0], 128), torch.randn(data.shape[0], 128))
                test_loss += nn.L1Loss()(output, target).item()
                _, predicted = torch.max(output, 1)
                correct += (predicted == target).sum().item()

                accuracy = correct / len(test_loader.dataset)
                print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * accuracy))

                # Main function

    def main():
        # Set device (GPU or CPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model and dataset
        model = get_model(embedding_dimension=128, patch_size=16)
        dataset = get_dataset(data_dir='/path/to/data', target_resolution=256)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        # Evaluate model
        evaluate(model, device, test_loader)

        if __name__ == '__main__':
            main()
