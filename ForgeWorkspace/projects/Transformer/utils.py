import os
import torch
import logging

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


def save_checkpoint(state, is_best, checkpoint_dir='.'):
    checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, os.path.join(checkpoint_dir, 'model_best.pth.tar'))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


class TrainingLogger(object):
    def __init__(self, log_file):
        self.log_file = log_file
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.handler = logging.FileHandler(log_file)
        self.handler.setLevel(logging.INFO)
        self.logger.addHandler(self.handler)

    def log(self, message):
        self.logger.info(message)

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
