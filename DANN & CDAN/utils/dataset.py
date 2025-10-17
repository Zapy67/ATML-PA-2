from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, Optional, Dict
import tqdm
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class OfficeHomeDataset(torch.utils.data.Dataset):    
    def __init__(self, root_dir, csv_file, domains, transform=None):
        self.df = pd.read_csv(csv_file)
        print(self.df.shape)
        print(self.df.head())

        # Convert Windows paths to Kaggle-compatible paths
        self.df['domain'] = self.df['name'].apply(lambda x: x.split('/')[2])
        self.df['image'] = self.df['name'].apply(
            lambda x: x.replace("D:/Dataset10072016", root_dir).strip()
        )
        self.df['label']  = self.df['name'].apply(lambda x: x.split('/')[3])
       
        self.df = self.df[self.df['domain'].isin(domains)].reset_index(drop=True)
        self.df = self.df[self.df['label'] != "Clock"].reset_index(drop=True)
        self.transform = transform
        
        self.classes = sorted(self.df['label'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.df.iloc[idx]
        path = row['image']
        label = row['label']
        
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, self.class_to_idx[label]

class RestWrapper(nn.Module):
    """Wrap the `truncate_from(resnet, 'layer3')` module so it returns (B, D)
       and exposes output_dim attribute expected by DANN class."""
    def __init__(self, rest_module, output_dim=2048):
        super().__init__()
        self.rest = rest_module
        self.output_dim = output_dim

    def forward(self, featmap):
        # featmap: (B, C, H, W)  -> rest likely includes layer4 + avgpool => (B,C,1,1)
        out = self.rest(featmap)
        # If rest returns (B, C, 1, 1) or (B, C), flatten to (B, D)
        if out.dim() == 4:
            out = torch.flatten(out, 1)
        elif out.dim() == 2:
            # already flattened
            pass
        else:
            # keep guard
            out = out.view(out.size(0), -1)
        return out

class FeatureTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_model, layer_name="layer3", device="cpu", batch_size=32, num_workers=2):
        self.device = device
        self.layer_name = layer_name

        # trunk: images -> layer3 feature maps (B, C, H, W)
        self.trunk = truncate_up_to(base_model, layer_name).to(device).eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        feats, labs = [], []
        with torch.inference_mode():
            for imgs, lbls in tqdm.tqdm(loader, desc=f"Precomputing up to {layer_name}"):
                imgs = imgs.to(device)
                fmaps = self.trunk(imgs)           # (B, C, H, W)
                feats.append(fmaps.cpu())          # keep on CPU
                labs.append(lbls)
        self.x = torch.cat(feats)   # shape (N, C, H, W) on CPU
        self.y = torch.cat(labs)    # shape (N,)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        # Return CPU tensors; DataLoader will collate them and move to device if needed.
        return self.x[idx], self.y[idx]

def truncate_up_to(resnet: nn.Module, layer_name: str) -> nn.Sequential:
    seq = nn.Sequential()
    for name, module in resnet.named_children():
        seq.add_module(name, module)
        if name == layer_name:
            break
    return seq

def truncate_from(resnet: nn.Module, layer_name: str) -> nn.Sequential:
    seen = False
    seq = nn.Sequential()
    for name, module in resnet.named_children():
        if seen and name != 'fc':    # exclude final fc
            seq.add_module(name, module)
        if name == layer_name:
            seen = True
    return seq

def freeze_until(resnet: nn.Module, layer_name: str):
    freeze = True
    for name, module in resnet.named_children():
        if freeze:
            for p in module.parameters():
                p.requires_grad = False
        if name == layer_name:
            freeze = False
