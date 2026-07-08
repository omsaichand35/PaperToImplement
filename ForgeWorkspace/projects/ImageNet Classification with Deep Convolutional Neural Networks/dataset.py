import os
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if self.train:
            self.data_dir = os.path.join(self.root, 'train')
        else:
            self.data_dir = os.path.join(self.root, 'val')

        self.classes = os.listdir(self.data_dir)
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self.make_dataset(self.data_dir)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = torchvision.load_image(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)

    def make_dataset(self, dir):
        images = []
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if fname.endswith('.JPEG'):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[root.split('/')[-1]])
                    images.append(item)
        return images


def load_data(root, batch_size=128, num_workers=4):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageNetDataset(root=root, train=True, transform=train_transform)
    val_dataset = ImageNetDataset(root=root, train=False, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=num_workers,
                                               pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=True)

    return train_loader, val_loader
