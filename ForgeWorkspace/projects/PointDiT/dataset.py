import torch
from torchvision import transforms
from PIL import Image
import numpy as np

# Define the dataset class
class PointDiTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.labels = []

        # Load the data
        for file in os.listdir(data_dir):
            if file.endswith('.jpg') or file.endswith('.png'):
                img = Image.open(os.path.join(data_dir, file))
                self.data.append(img)
                self.labels.append(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        label = self.labels[index]

        # Apply the transformation
        if self.transform:
            img = self.transform(img)

            # Normalize the point map
            img = self.normalize_point_map(img)

            return img, label

    def normalize_point_map(self, img):
        # Standardize the point maps by subtracting centroid and dividing by scalar scale factor
        centroid = np.mean(img, axis=(0, 1))
        scale_factor = np.std(img, axis=(0, 1))
        img = (img - centroid) / scale_factor
        return img

    # Define the data preprocessing pipeline
transform = transforms.Compose([
transforms.Resize((512, 512)),
transforms.CenterCrop(512),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    # Create the dataset
dataset = PointDiTDataset('data', transform=transform)
