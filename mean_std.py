import torchvision
from torchvision import transforms


# calculate mean, std


transform = transforms.Compose([
    transforms.ToTensor()
])
dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
mean = dataset.data.mean(axis=(0,1,2))
std = dataset.data.std(axis=(0,1,2))