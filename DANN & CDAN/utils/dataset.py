from torch.utils.data import Dataset
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class DeepLakeWrapper(Dataset):
    def __init__(self, ds, domain_label=None, img_size=224):
        self.ds = ds
        self.domain_label = domain_label
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[int(idx)]
        img = sample['images'].numpy().astype('uint8')
        label = int(sample['domain_objects'].numpy()[0])
        img = self.transform(img)  # ensure consistent tensor size

        if self.domain_label is not None:
            return img, label, self.domain_label
        return img, label