import os
import torch
import torch.nn as nn
import torch.optim as optim


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    """Saves a model and training parameters at checkpoint + 'last.pth.tar'.
    If is_best==True, also saves checkpoint + 'model_best.pth.tar'

    Args:
        state: (dictionary) contains model's state_dict, may contain other keys such as epoch, optimizer_state_dict
        is_best: (bool) True if it is the best model seen so far
        checkpoint_dir: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint_dir, 'last.pth.tar')
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads a model and training parameters.

    Args:
        checkpoint_path: (string) folder where parameters are saved
        model: torch.nn.Module
        optimizer: torch.optim

    Returns:
        model: torch.nn.Module
        optimizer: torch.optim
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer