import torch
import torchvision
import torchvision.transforms as transforms

# Define data augmentation transforms
transform = transforms.Compose([
 transforms.Resize((256, 256)),
 transforms.CenterCrop(256),
 transforms.ToTensor(),
 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load SceneNet-RGBD dataset
train_dataset = torchvision.datasets.SceneNetRGBD(root='./data', train=True, download=True, transform=transform)

# Create data loader
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
