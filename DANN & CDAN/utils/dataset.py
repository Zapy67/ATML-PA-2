from torch.utils.data import Dataset, Subset
from torchvision import transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Tuple, Optional, Dict

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

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