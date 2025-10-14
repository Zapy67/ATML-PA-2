from torch.utils.data import Dataset
from torchvision import transforms, models
import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import tqdm


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
    
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def filter_domain(ds, domain_ids):
    indices = [i for i in range(len(ds)) if int(ds[i]["domain_objects"].numpy()[0]) in domain_ids]
    return ds[indices]

class OfficeHomeGroups(Dataset):
    """
    Automatically groups Office-Home (or similar) datasets using Domain × Cluster logic.
    
    Args:
        base_dataset: a PyTorch Dataset returning (img, label, domain_id)
        num_domains: total number of domains (e.g. 4 for Office-Home)
        clusters_per_domain: number of clusters per domain (e.g. 3–5)
        backbone_name: pretrained backbone for feature extraction ('resnet50', 'vit_b_16', etc.)
        device: torch device ('cuda' or 'cpu')
        transform: optional preprocessing for feature extraction
    """
    def __init__(
        self,
        base_dataset: Dataset,
        num_domains: int = 4,
        clusters_per_domain: int = 5,
        backbone_name: str = "resnet50",
        device: str = "cuda",
        transform=None,
        verbose=True,
    ):
        self.base_dataset = base_dataset
        self.num_domains = num_domains
        self.K = clusters_per_domain
        self.device = device
        self.verbose = verbose

        # Default transform for embedding extraction
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Initialize backbone
        if verbose:
            print(f"[OfficeHomeGroups] Loading pretrained {backbone_name} backbone...")
        self.backbone = self._load_backbone().to(device)
        self.backbone.eval()

        # Step 1: extract embeddings
        if verbose:
            print("[OfficeHomeGroups] Extracting embeddings...")
        feats, domains = self._extract_embeddings()

        # Step 2: cluster embeddings per domain
        if verbose:
            print("[OfficeHomeGroups] Clustering embeddings per domain...")
        self.group_ids = self._cluster_domains(feats, domains)

        if verbose:
            print(f"[OfficeHomeGroups] Created {self.num_domains * self.K} total groups.")

    # ----------------------------
    # Backbone setup
    # ----------------------------
    def _load_backbone(self):

        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        return torch.nn.Sequential(*list(model.children())[:-1])

    # ----------------------------
    # Embedding extraction
    # ----------------------------
    def _extract_embeddings(self):
        feats, domains = [], []
        loader = torch.utils.data.DataLoader(self.base_dataset, batch_size=64, shuffle=False, num_workers=2)
        for imgs, _, domain in tqdm(loader, desc="Extracting features", disable=not self.verbose):
            imgs = imgs.to(self.device)
            with torch.inference_mode():
                f = self.backbone(imgs).squeeze()
                if f.ndim == 1:
                    f = f.unsqueeze(0)
            feats.append(f.cpu())
            domains.append(domain.cpu())
        return torch.cat(feats), torch.cat(domains)

    # ----------------------------
    # Clustering
    # ----------------------------
    def _cluster_domains(self, feats, domains):
        group_ids = torch.zeros(len(domains), dtype=torch.long)
        for d in range(self.num_domains):
            idx = (domains == d).nonzero(as_tuple=True)[0]
            if len(idx) == 0:
                continue
            sub_feats = feats[idx].numpy()
            km = KMeans(n_clusters=self.K, random_state=42, n_init="auto").fit(sub_feats)
            group_ids[idx] = torch.tensor(km.labels_ + d * self.K)
        return group_ids

    # ----------------------------
    # Dataset interface
    # ----------------------------
    def __getitem__(self, idx):
        img, label, domain = self.base_dataset[idx]
        group = int(self.group_ids[idx])
        return img, label, domain, group

    def __len__(self):
        return len(self.base_dataset)

def visualize_groups(ds, domain_map, max_per_domain=5):
    import matplotlib.pyplot as plt
    import random

    plt.figure(figsize=(15, 6))
    seen = set()
    count = 1
    for d in range(len(domain_map)):
        idxs = [i for i, (_, _, dom, _) in enumerate(ds) if dom == d]
        random.shuffle(idxs)
        for i in idxs[:max_per_domain]:
            img, label, dom, group = ds[i]
            plt.subplot(len(domain_map), max_per_domain, count)
            plt.imshow(np.asarray(img))
            plt.title(f"{domain_map[dom]}\nG{group}")
            plt.axis("off")
            count += 1
    plt.tight_layout()
    plt.show()
