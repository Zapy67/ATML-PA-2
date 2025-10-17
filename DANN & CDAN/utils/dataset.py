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

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

class OfficeHomeDataset(torch.utils.data.Dataset):    
    def __init__(self, root_dir, csv_file, domains, transform=None):
        self.df = pd.read_csv(csv_file)       
        # Convert Windows paths to Kaggle-compatible paths
        self.df['name'] = self.df['name'].apply(
            lambda x: x.replace("D:/Dataset10072016", "/kaggle/input/officehome/OfficeHomeDataset_10072016")
        )
        self.df['domain'] = self.df['name'].apply(lambda x: x.split('/')[2])
        self.df['image']  = self.df['name'].apply(lambda x: os.path.join(root_dir, x.split('/', 2)[-1].strip()))
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

def _truncate_resnet(model, layer_name):
        layers = nn.Sequential()
        for name, module in model.named_children():
            layers.add_module( name, module)
            if name == layer_name:
                break
        return layers

def _truncate_resnet_from(model, layer_name):
        seen = False
        layers = nn.Sequential()
        for name, module in model.named_children():
            if seen:
                layers.add_module( name, module)
            if name == layer_name:
                seen = True
        return layers

class FeatureTensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_model, layer_name="layer3", device="cpu", batch_size=32):
        self.dataset = dataset
        self.device = device
        self.layer_name = layer_name

        base_model.eval()
        self.feature_extractor = _truncate_resnet(base_model, layer_name).to(device)
        self.feature_extractor.eval()        
        self.x, self.y = self._precompute_features(dataset, batch_size)

        self.truncated_model = _truncate_resnet_from(base_model, layer_name).to(device)

    
    def _precompute_features(self, dataset, batch_size):
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        feats, labels = [], []

        with torch.inference_mode():
            for imgs, lbls in tqdm(loader, desc=f"Precomputing up to {self.layer_name}"):
                imgs = imgs.to(self.device)
                outputs = self.feature_extractor(imgs)
                feats.append(outputs.cpu())
                labels.append(lbls)

        x = torch.cat(feats)
        y = torch.cat(labels)
        return x, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class DeepLakeWrapper(Dataset):
    """
    Wrapper for OfficeHome DeepLake dataset.

    - Optionally pass domain_map (e.g. {0: "RealWorld", 1: "Product", 2: "Art", 3: "Clipart"})
      to get human-readable names if needed.
    """

    def __init__(
        self,
        ds,
        img_size: int = 224,
        domain_map: Optional[Dict[int, str]] = None
    ):
        self.ds = ds
        self.domain_map = domain_map
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.ds)
    
    def _to_np(self, x):
        # Convert tensor-like to numpy safely
        try:
            if hasattr(x, "numpy"):
                return x.numpy()
        except Exception:
            pass
        return np.asarray(x)

    def __getitem__(self, idx):
        sample = self.ds[int(idx)]

        # --- image ---
        if 'images' in sample:
            img_arr = self._to_np(sample['images'])
        elif 'image' in sample:
            img_arr = self._to_np(sample['image'])
        else:
            raise KeyError("Expected sample to have 'images' or 'image' key.")

        # normalize shape: if CHW -> HWC; if grayscale -> stack to 3 channels
        img_arr = np.asarray(img_arr)
        if img_arr.ndim == 3 and img_arr.shape[0] in (1,3) and img_arr.shape[0] != img_arr.shape[2]:
            # CHW -> HWC
            img_arr = np.transpose(img_arr, (1, 2, 0))
        if img_arr.ndim == 2:
            img_arr = np.stack([img_arr]*3, axis=-1)
        img_arr = img_arr.astype(np.uint8)

        img_tensor = self.transform(img_arr)

        # --- label ---
        if 'domain_objects' in sample:
            lbl = self._to_np(sample['domain_objects'])
        elif 'label' in sample:
            lbl = self._to_np(sample['label'])
        elif 'target' in sample:
            lbl = self._to_np(sample['target'])
        else:
            raise KeyError("Expected sample to have 'domain_objects' or 'label'/'target' key for class label.")
        # extract scalar
        lbl = int(np.asarray(lbl).reshape(-1)[0])

        # --- domain id (OfficeHome specific) ---
        if 'domain_categories' not in sample:
            raise KeyError("Expected sample to have 'domain_categories' key for domain id (OfficeHome).")
        dom = self._to_np(sample['domain_categories'])
        dom = int(np.asarray(dom).reshape(-1)[0])

        return img_tensor, lbl, dom