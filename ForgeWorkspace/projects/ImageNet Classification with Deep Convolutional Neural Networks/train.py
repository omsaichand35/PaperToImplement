import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import time
import math
import numpy as np
import argparse
from models import AlexNet
from dataset import load_data
from utils import AverageMeter, save_checkpoint, load_checkpoint

# Hyperparameters
BATCH_SIZE = 128
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005
LEARNING_RATE = 0.01
EPOCHS = 10

# Define the device (GPU or CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the dataset
train_loader, val_loader = load_data(root="./data", batch_size=BATCH_SIZE)

# Initialize the model, optimizer, and loss function
model = AlexNet()
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(EPOCHS):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the model parameters
        optimizer.step()

        # Measure accuracy and record loss
        prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

    print(f'Epoch {epoch+1}, Loss: {losses.avg:.4f}, Top1 Acc: {top1.avg:.2f}%, Top5 Acc: {top5.avg:.2f}%')

    # Save the model checkpoint
    save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc1': top1.avg, 'optimizer': optimizer.state_dict()}, is_best=False, checkpoint_dir='./checkpoints')

# Evaluation
def evaluate(model, val_loader):
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

    print(f'Test Acc: Top1 {top1.avg:.2f}%, Top5 {top5.avg:.2f}%')

# Accuracy function
def accuracy(outputs, labels, topk=(1,)):
    maxk = max(topk)
    batch_size = labels.size(0)

    _, pred = outputs.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
