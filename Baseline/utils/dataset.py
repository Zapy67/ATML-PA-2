from PIL import Image
import torch
from torch import nn
import pandas as pd
import os
from tqdm import tqdm

class OfficeHomeDataset(torch.utils.data.Dataset):    
    def __init__(self, root_dir, csv_file, domains, transform=None):
        self.df = pd.read_csv(csv_file)       
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
                if name == "fc":
                    layers.add_module('avgpool', model.avgpool)
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