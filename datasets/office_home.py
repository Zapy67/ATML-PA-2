from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
import pandas as pd
import os
import zipfile

SOURCES = ["Art", "Product", "Real World"]
TARGET = ["Clipart"]

class OfficeHomeDataset(Dataset):    
    """
    Wraps Office Home dataset into a torch Dataset Object
    """
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

class OfficeHomeData:
    def __init__(self, root_dir="./data/OfficeHomeDataset", csv_file="ImageInfo.csv", zip_path="./data/OfficeHomeDataset_10072016.zip"):
        """
        Wrapper to handle Office-Home dataset:
        - Extracts data if not already extracted
        - Provides PyTorch Dataset/DataLoader objects
        """
        self.root_dir = root_dir
        self.csv_file = os.path.join(root_dir, csv_file)
        self.zip_path = zip_path

        self._ensure_extracted()
        self._ensure_csv_exists()

    def _ensure_extracted(self):
        """
        Checks if dataset folders exist; if not, extracts the zip file.
        """
        if not os.path.exists(os.path.join(self.root_dir, "Art")):
            if not os.path.exists(self.zip_path):
                raise FileNotFoundError(f"Dataset zip not found at {self.zip_path}")
            
            print("Extracting Office-Home dataset...")
            os.makedirs(self.root_dir, exist_ok=True)
            with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(self.root_dir))
            print("Extraction complete.")
        else:
            print("Dataset already extracted at", self.root_dir)

    def _ensure_csv_exists(self):
        """
        Generates ImageInfo.csv if not present by scanning all image files.
        """
        if os.path.exists(self.csv_file):
            print("ImageInfo.csv already exists.")
            return

        print("Generating ImageInfo.csv ...")
        image_entries = []
        for domain in ["Art", "Clipart", "Product", "Real World"]:
            domain_path = os.path.join(self.root_dir, domain)
            if not os.path.isdir(domain_path):
                continue
            for label in os.listdir(domain_path):
                label_dir = os.path.join(domain_path, label)
                if not os.path.isdir(label_dir):
                    continue
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        rel_path = f"{self.root_dir}/{domain}/{label}/{fname}"
                        # match format like original dataset 'name' column
                        image_entries.append({'name': rel_path})

        df = pd.DataFrame(image_entries)
        df.to_csv(self.csv_file, index=False)
        print(f"Created CSV with {len(df)} entries at {self.csv_file}")

    def get_dataset(self, domains, transform=None):
        """
        Returns a torch.utils.data.Dataset for specified domains.
        """
        if transform is None:
            transform = ResNet50_Weights.IMAGENET1K_V1.transforms()
        return OfficeHomeDataset(self.root_dir, self.csv_file, domains, transform)

    def get_dataloader(self, domains, batch_size=32, shuffle=True, num_workers=2, transform=None):
        """
        Returns a DataLoader for the given domains.
        """
        dataset = self.get_dataset(domains, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return loader

if __name__ == "__main__":
    office = OfficeHomeData(
        root_dir="./data/OfficeHomeDataset",
        zip_path="./data/OfficeHomeDataset_10072016.zip"
    )

    sources = SOURCES
    target = TARGET

    source_loader = office.get_dataloader(sources, batch_size=32)
    target_loader = office.get_dataloader(target, batch_size=32)

    print(f"Source dataset size: {len(source_loader.dataset)}")
    print(f"Target dataset size: {len(target_loader.dataset)}")
