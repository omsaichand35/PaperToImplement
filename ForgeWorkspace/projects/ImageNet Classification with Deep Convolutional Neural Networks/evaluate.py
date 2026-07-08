from dataset import load_data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


def evaluate(model, device, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet()
    model.to(device)
    _, val_loader = load_data(root='./data', batch_size=128, num_workers=4)
    evaluate(model, device, val_loader)


if __name__ == '__main__':
    main()
