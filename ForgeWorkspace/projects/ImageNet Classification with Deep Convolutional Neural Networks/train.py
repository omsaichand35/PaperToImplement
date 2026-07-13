import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from models import AlexNet
from dataset import get_dataset, get_data_loader
from utils import AverageMeter, save_checkpoint, load_checkpoint


# Hyperparameters
batch_size = 128
learning_rate = 0.01
weight_decay = 0.0005
num_cycles = 90


# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load the dataset and create data loaders
root = './data'
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = get_dataset(root, train=True, transform=transform)
train_loader = get_data_loader(train_dataset, batch_size)

val_dataset = get_dataset(root, train=False, transform=transform)
val_loader = get_data_loader(val_dataset, batch_size, shuffle=False)


# Initialize the model, optimizer, and loss function
model = AlexNet(num_classes=1000)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.CrossEntropyLoss()


# Train the model
def train(model, device, loader, optimizer, loss_fn):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(prec1.item(), data.size(0))
        top5.update(prec5.item(), data.size(0))

    print(f'Train Loss: {losses.avg:.4f} | Train Acc@1: {top1.avg:.2f}% | Train Acc@5: {top5.avg:.2f}%')


# Evaluate the model
def evaluate(model, device, loader, loss_fn):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_fn(output, target)

            # Measure accuracy and record loss
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(prec1.item(), data.size(0))
            top5.update(prec5.item(), data.size(0))

    print(f'Val Loss: {losses.avg:.4f} | Val Acc@1: {top1.avg:.2f}% | Val Acc@5: {top5.avg:.2f}%')


# Train the model for 90 cycles
best_acc1 = 0
for epoch in range(num_cycles):
    train(model, device, train_loader, optimizer, loss_fn)
    evaluate(model, device, val_loader, loss_fn)

    # Save checkpoint
    is_best = False
    if top1.avg > best_acc1:
        best_acc1 = top1.avg
        is_best = True
    save_checkpoint({'epoch': epoch + 1,
                     'state_dict': model.state_dict(),
                     'best_acc1': best_acc1,
                     'optimizer': optimizer.state_dict()},
                    is_best, checkpoint_dir='./checkpoints')


# Helper function to calculate accuracy
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
