import torch
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pandas as pd
import os
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader
import random
import math

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

def prepare_dataframe(csv_file, root_dir, domains, exclude_label="Clock"):
        df = pd.read_csv(csv_file)
        # parse same way as original scripts expect
        df['domain'] = df['name'].apply(lambda x: x.split('/') [2])
        df['image'] = df['name'].apply(lambda x: x.replace("D:/Dataset10072016", root_dir).strip())
        df['label'] = df['name'].apply(lambda x: x.split('/') [3])
        df = df[df['domain'].isin(domains)].reset_index(drop=True)
        if exclude_label is not None:
            df = df[df['label'] != exclude_label].reset_index(drop=True)
        classes = sorted(df['label'].unique())
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return df, classes, class_to_idx

class OfficeHomeDataset(torch.utils.data.Dataset):    
    def __init__(self, root_dir, csv_file, domains, transform=None):
        self.df, self.classes, self.class_to_idx = prepare_dataframe(csv_file, root_dir, domains)
        self.transform = transform

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

class OfficeHomeSubset(torch.utils.data.Dataset):
    """Subset dataset backed by a dataframe + indices. Returns (image, label_idx)."""
    def __init__(self, df, indices, class_to_idx, transform=None, img_size=224):
        self.df = df
        self.indices = list(indices)
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.df.iloc[self.indices[idx]]
        path = row['image']
        label = row['label']
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
        if self.transform:
            image = self.transform(image)
        return image, self.class_to_idx[label]


def split_indices_by_domain(df, domains, seed=42, test_frac=0.1):
    domain_to_id = {d: i for i, d in enumerate(domains)}
    domain_to_indices = {domain_to_id[d]: [] for d in domains}
    for idx, row in df.iterrows():
        dom = row['domain']
        if dom in domain_to_id:
            domain_to_indices[domain_to_id[dom]].append(idx)
    rng = random.Random(seed)
    domain_to_train = {}
    domain_to_test = {}
    for d, idxs in domain_to_indices.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n_test = int(len(idxs) * test_frac)
        domain_to_test[d] = idxs[:n_test]
        domain_to_train[d] = idxs[n_test:]
    return domain_to_train, domain_to_test


def make_domain_subdatasets(csv_file, root_dir, domains, train_transform, test_transform,
                            test_frac=0.1, seed=42, img_size=224, sub_bs=8, num_workers=2):
    """Create per-domain train/test subsets and return loaders/datasets used by trainers.

    Returns:
      domain_train_datasets, domain_test_datasets, class_to_idx, steps_per_epoch, test_items
    """
    df, classes, class_to_idx = prepare_dataframe(csv_file, root_dir, domains)
    domain_train_idxs, domain_test_idxs = split_indices_by_domain(df, domains, seed=seed, test_frac=test_frac)

    domain_train_datasets = []
    domain_test_datasets = []
    for d in range(len(domains)):
        train_d = OfficeHomeSubset(df, domain_train_idxs[d], class_to_idx, transform=train_transform, img_size=img_size)
        test_d = OfficeHomeSubset(df, domain_test_idxs[d], class_to_idx, transform=test_transform, img_size=img_size)
        domain_train_datasets.append(train_d)
        domain_test_datasets.append(test_d)

    # test items list (tensor, label_idx, domain_id) built by fetching from vd (which applies test_transform)
    test_items = []
    for d, vd in enumerate(domain_test_datasets):
        for i in range(len(vd)):
            img_t, label = vd[i]
            test_items.append((img_t, int(label), d))

    max_len = max(len(d) for d in domain_train_datasets)
    steps_per_epoch = math.ceil(max_len / float(max(1, sub_bs)))
    return domain_train_datasets, domain_test_datasets, class_to_idx, steps_per_epoch, test_items, classes


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

