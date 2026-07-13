import torch
import torchvision
import torchvision.transforms as transforms


class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]

        if self.transform is not None:
            image = self.transform(image)

            if self.target_transform is not None:
                target = self.target_transform(target)

                return image, target

    def __len__(self):
        return len(self.data)

    def load_data(self):
        if self.train:
            self.data, self.targets = torchvision.datasets.ImageNet(root=self.root, split='train', download=self.download)
        else:
            self.data, self.targets = torchvision.datasets.ImageNet(root=self.root, split='val', download=self.download)

    def __init__(self):
        self.load_data()


    def get_dataset(root, train=True, transform=None, target_transform=None, download=False):
        dataset = ImageNetDataset(root, train, transform, target_transform, download)
        return dataset


    def get_data_loader(dataset, batch_size, shuffle=True, num_workers=4):
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader


def main():
    root = './data'
    batch_size = 128

    transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageNetDataset(root, train=True, transform=transform)
    data_loader = ImageNetDataset(dataset, batch_size)

    for images, labels in data_loader:
        print(images.shape, labels.shape)
        break

if __name__ == '__main__':
    main()