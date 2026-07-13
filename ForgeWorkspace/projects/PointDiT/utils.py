import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    checkpoint_path = checkpoint_dir + '/checkpoint.pth.tar'
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, checkpoint_dir + '/model_best.pth.tar')


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'], checkpoint['best_acc1']


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count