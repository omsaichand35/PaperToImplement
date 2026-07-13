import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from models import AlexNet
from dataset import get_dataset, get_data_loader
from utils import AverageMeter


def evaluate(model, device, data_loader):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            top1.update(correct / len(labels), len(labels))
            _, predicted = outputs.topk(5, 1, True, True)
            correct = predicted.eq(labels.view(-1, 1).expand_as(predicted))
            correct = correct.float().sum().item()
            top5.update(correct / len(labels), len(labels))

    return top1.avg, top5.avg


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AlexNet(num_classes=1000)
    model.to(device)

    root = './data'
    batch_size = 128

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = get_dataset(root, train=False, transform=transform)
    data_loader = get_data_loader(dataset, batch_size, shuffle=False)

    top1, top5 = evaluate(model, device, data_loader)
    print(f'Top-1 accuracy: {top1:.2f}%')
    print(f'Top-5 accuracy: {top5:.2f}%')

if __name__ == '__main__':
    main()