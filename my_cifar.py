import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import Dataset
class MyCIFAR10(Dataset):
    
    def __init__(self, train=True):
        super().__init__()
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform = transforms.ToTensor()
        self.train = train
        self.dst = datasets.CIFAR10('data', train=train, download=True, transform=transform)
        
    def __getitem__(self, idx):
        x_star = self.dst[idx][0]
        label =  self.dst[idx][1]
        x = x_star
        x[1,:,:] = 0 
        x[2,:,:] = 0 
        if self.train:
            return x, label, x_star
        else:
            return x, label
     
    def __len__(self):
        return len(self.dst)