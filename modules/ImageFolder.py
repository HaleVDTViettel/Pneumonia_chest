import torch
import albumentations as A
import numpy as np
from torchvision import datasets, transforms

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self,h , root, transform=None, is_valid_file=None):
        self.h = h
        self.dataset = datasets.ImageFolder(root, is_valid_file=is_valid_file)
        self.transform = transform
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.h['albumentations']:
            image_transform = self.transform(image=np.array(image))["image"] / 255.0
        else:
            image_transform = self.transform(image)
        return image_transform, label

    def __len__(self):
        return len(self.dataset)