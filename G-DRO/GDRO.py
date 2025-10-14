import math
import os
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image

import deeplake


# ---------------------------
# Config / hyperparams
# ---------------------------
DATA_HUB = "hub://activeloop/office-home-domain-adaptation"
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 65
NUM_DOMAINS = 4   # Art, Clipart, Product, RealWorld (these are groups)
BATCH_SIZE = 64   # total batch size = sub_bs * NUM_DOMAINS
NUM_WORKERS = 4
IMG_SIZE = 224

LR = 3e-4
WEIGHT_DECAY = 1e-3   # paper suggests stronger L2 may help
ETA_Q = 0.5           # multiplicative weights step (tune: 0.01-1.0)
EMA_ALPHA = 0.2       # EMA smoothing for running group loss
NUM_EPOCHS = 20
PATIENCE = 5          # early stopping on worst-group val
MODEL_OUT = "g_dro_officehome_balanced.pt"

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# sanity check
assert BATCH_SIZE >= NUM_DOMAINS, "BATCH_SIZE must be >= NUM_DOMAINS"
sub_bs = max(1, BATCH_SIZE // NUM_DOMAINS)

# ---------------------------
# Determinism
# ---------------------------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Helper: robust extraction of scalars from deeplake samples
# ---------------------------
def _to_int_scalar(x):
    import numpy as _np
    a = _np.array(x)
    if a.size == 0:
        return None
    return int(a.ravel()[0])

# ---------------------------
# Domain-specific Deep Lake subset dataset
# returns (img_tensor, label_int)  - group is known by parent
# ---------------------------
class DomainDeepLakeSubset(Dataset):
    def __init__(self, deeplake_ds, indices, transform=None):
        """
        deeplake_ds: loaded Deep Lake dataset
        indices: list of global indices into deeplake_ds that belong to this domain
        transform: torchvision transform to apply to PIL image
        """
        self.ds = deeplake_ds
        self.indices = list(indices)
        self.transform = transform or transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=MEAN, std=STD)])
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = int(self.indices[idx])
        sample = self.ds[global_idx]
        img_np = sample['images'].numpy()
        # normalize to uint8 HWC if needed
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
        img = Image.fromarray(img_np)
        img_t = self.transform(img)

        # label extraction: prefer domain_objects for class labels (65 unique values)
        if 'domain_objects' in self.ds.tensors and np.unique(self.ds['domain_objects'][:].numpy()).size >= NUM_CLASSES:
            label = _to_int_scalar(sample['domain_objects'].numpy())
        elif 'labels' in self.ds.tensors:
            label = _to_int_scalar(sample['labels'].numpy())
        else:
            # fallback to domain_objects as label
            label = _to_int_scalar(sample['domain_objects'].numpy())
        return img_t, int(label)
    
# ---------------------------
# Utility functions
# ---------------------------
def split_indices_by_domain(ds, val_frac=0.1, seed=SEED):
    """
    Returns dict:
      domain_to_train_indices[domain] = list(...)
      domain_to_val_indices[domain]   = list(...)
    """
    domain_to_indices = {d: [] for d in range(NUM_DOMAINS)}
    # gather indices per domain
    for i in range(len(ds)):
        try:
            dom = _to_int_scalar(ds[i]['domain_categories'].numpy())
        except Exception:
            # fallback: some variants store domain in domain_objects
            dom = _to_int_scalar(ds[i]['domain_objects'].numpy())
        if dom is None:
            dom = 0
        domain_to_indices[int(dom)].append(i)

    domain_to_train = {}
    domain_to_val = {}
    rng = random.Random(seed)
    for d, idxs in domain_to_indices.items():
        rng.shuffle(idxs)
        n_val = int(len(idxs) * val_frac)
        domain_to_val[d] = idxs[:n_val]
        domain_to_train[d] = idxs[n_val:]
    return domain_to_train, domain_to_val

def compute_group_counts_loaders(domain_loaders):
    counts = {}
    for i, loader in enumerate(domain_loaders):
        cnt = 0
        for x,y in loader:
            cnt += x.size(0)
        counts[i] = cnt
    return counts

def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='none')
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    losses = []
    with torch.no_grad():
        for imgs, labels, groups in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1)
            for i in range(labels.size(0)):
                gi = int(groups[i].item())
                group_total[gi] += 1
                if int(preds[i].cpu().item()) == int(labels[i].cpu().item()):
                    group_correct[gi] += 1
            losses.append(criterion(out, labels).cpu())
    per_group_acc = {}
    for g in range(NUM_DOMAINS):
        if group_total[g] > 0:
            per_group_acc[g] = group_correct[g] / group_total[g]
        else:
            per_group_acc[g] = None
    all_losses = torch.cat(losses).numpy() if len(losses) > 0 else np.array([])
    avg_loss = float(all_losses.mean()) if all_losses.size > 0 else None
    worst_group_acc = min([a for a in per_group_acc.values() if a is not None])
    return {'per_group_acc': per_group_acc, 'avg_loss': avg_loss, 'worst_group_acc': worst_group_acc}

# ---------------------------
# Load dataset & split per-domain
# ---------------------------
print("Loading dataset from Activeloop hub...")
ds = deeplake.load(DATA_HUB)

domain_to_train_idxs, domain_to_val_idxs = split_indices_by_domain(ds, val_frac=0.1, seed=SEED)
for d in range(NUM_DOMAINS):
    print(f"Domain {d}: train={len(domain_to_train_idxs[d])}, val={len(domain_to_val_idxs[d])}")

# ---------------------------
# Build domain-specific PyTorch Datasets and DataLoaders
# ---------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

domain_train_datasets = []
domain_val_datasets = []
for d in range(NUM_DOMAINS):
    td = DomainDeepLakeSubset(ds, domain_to_train_idxs[d], transform=train_transform)
    vd = DomainDeepLakeSubset(ds, domain_to_val_idxs[d], transform=val_transform)
    domain_train_datasets.append(td)
    domain_val_datasets.append(vd)

# DataLoaders per domain (balanced sub-batch size)
domain_train_loaders = [
    DataLoader(td, batch_size=sub_bs, shuffle=True, drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)
    for td in domain_train_datasets
]

# simpler: build val loader that yields (img, label, group) directly by iterating per-domain and tagging
val_items = []
for d, vd in enumerate(domain_val_datasets):
    for i in range(len(vd)):
        img_t, label = vd[i]
        val_items.append((img_t, int(label), d))
val_loader = DataLoader(val_items, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Train dataset sizes:", [len(x) for x in domain_train_datasets])
print("Val total samples:", len(val_items))

# ---------------------------
# Model, optimizer, criterion
# ---------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss(reduction='none')  # per-sample

# ---------------------------
# Initialize G-DRO state
# ---------------------------
num_groups = NUM_DOMAINS
q = torch.ones(num_groups, device=DEVICE) / float(num_groups)  # adversarial weights
running_group_loss = torch.zeros(num_groups, device=DEVICE)   # EMA per-group loss
group_seen_counts = torch.zeros(num_groups, device=DEVICE)

best_worst = -1.0
patience_counter = 0

# Steps per epoch: ensure each epoch yields samples until the largest domain dataset is consumed
max_len = max(len(d) for d in domain_train_datasets)
steps_per_epoch = math.ceil(max_len / sub_bs)
print("steps_per_epoch:", steps_per_epoch, "sub_bs:", sub_bs)


loaders = domain_train_loaders
iters = [iter(dl) for dl in loaders]


# ---------------------------
# Training loop (balanced per-domain sampling)
# ---------------------------
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_losses = []
    with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{NUM_EPOCHS}") as pbar:
        for step in pbar:
            # --- Fetch one batch from each domain ---
            batches = []
            for i, it in enumerate(iters):
                try:
                    batch = next(it)
                except StopIteration:
                    # Reset iterator when exhausted
                    iters[i] = iter(loaders[i])
                    batch = next(iters[i])
                batches.append(batch)

            # --- Unpack domain batches ---
            imgs_art, labels_art = batches[0]
            imgs_prod, labels_prod = batches[1]
            imgs_real, labels_real = batches[2]
            imgs_clip, labels_clip = batches[3]

            # --- Merge all domains into one batch ---
            imgs = torch.cat([imgs_art, imgs_prod, imgs_real, imgs_clip], dim=0).to(DEVICE)
            labels = torch.cat([labels_art, labels_prod, labels_real, labels_clip], dim=0).to(DEVICE)

            groups = torch.cat([
                torch.full((imgs_art.size(0),), 0, dtype=torch.long),
                torch.full((imgs_prod.size(0),), 1, dtype=torch.long),
                torch.full((imgs_real.size(0),), 2, dtype=torch.long),
                torch.full((imgs_clip.size(0),), 3, dtype=torch.long)
            ], dim=0).to(DEVICE)

            logits = model(imgs)
            per_sample = criterion(logits, labels)  # (N,)

            # compute mean loss per domain present in the batch (all present by design)
            batch_group_losses = torch.zeros(num_groups, device=DEVICE)
            present = torch.zeros(num_groups, dtype=torch.bool, device=DEVICE)
            for g_id in range(num_groups):
                mask = (groups == g_id)
                if mask.sum() == 0:
                    continue
                lg = per_sample[mask].mean().detach()
                batch_group_losses[g_id] = lg
                present[g_id] = True
                # update running EMA
                running_group_loss[g_id] = (1.0 - EMA_ALPHA) * running_group_loss[g_id] + EMA_ALPHA * lg
                group_seen_counts[g_id] += 1

            # For missing groups (shouldn't happen here because balanced), fallback to running estimate
            used_group_losses = batch_group_losses.clone()
            for gi in range(num_groups):
                if not present[gi]:
                    if group_seen_counts[gi] > 0:
                        used_group_losses[gi] = running_group_loss[gi].detach()
                    else:
                        used_group_losses[gi] = torch.tensor(0.0, device=DEVICE)

            # multiplicative weights update (exponentiated gradient)
            q_np = (q * torch.exp(ETA_Q * used_group_losses.detach()))
            q = q_np / (q_np.sum() + 1e-12)

            # per-sample weights and loss
            w_i = q[groups]                # (N,)
            w_i = w_i / (w_i.sum() + 1e-12)
            loss = (w_i * per_sample).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f"{np.mean(epoch_losses):.4f}", 'worst_q': int(torch.argmax(q).item())})

    # End of epoch evaluation on validation (val_loader yields (img, label, group))
    # Build val batches manually from val_items (we constructed val_items earlier)
    model.eval()
    # We built val_loader as DataLoader(val_items, ...), where each entry is (img_t, label, domain)
    val_stats = {'per_group_acc': {}, 'avg_loss': None, 'worst_group_acc': None}
    # Build a proper loader for evaluation: convert val_items to tensors per batch:
    val_imgs, val_labels, val_groups = [], [], []
    for it in range(0, len(val_items), BATCH_SIZE):
        batch = val_items[it:it+BATCH_SIZE]
        imgs_b = torch.stack([b[0] for b in batch]).to(DEVICE)
        lbls_b = torch.tensor([b[1] for b in batch], dtype=torch.long).to(DEVICE)
        groups_b = torch.tensor([b[2] for b in batch], dtype=torch.long)
        val_imgs.append(imgs_b); val_labels.append(lbls_b); val_groups.append(groups_b)

    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    with torch.no_grad():
        for imgs_b, lbls_b, groups_b in zip(val_imgs, val_labels, val_groups):
            out = model(imgs_b)
            preds = out.argmax(dim=1).cpu().numpy()
            for i in range(len(preds)):
                gi = int(groups_b[i].item())
                group_total[gi] += 1
                if int(preds[i]) == int(lbls_b[i].cpu().item()):
                    group_correct[gi] += 1

    per_group_acc = {}
    for g in range(NUM_DOMAINS):
        per_group_acc[g] = (group_correct[g] / group_total[g]) if group_total[g] > 0 else None
    worst_acc = min([a for a in per_group_acc.values() if a is not None])
    print(f"Epoch {epoch} validation: worst_group_acc={worst_acc:.4f}, per_group={per_group_acc}")

    if worst_acc is not None and worst_acc > best_worst:
        best_worst = worst_acc
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'q': q.cpu().numpy()}, MODEL_OUT)
        print(f"  Saved best model (worst={best_worst:.4f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping: no improvement in worst-group val for {PATIENCE} epochs.")
            break

print("Training finished. Best worst-group acc:", best_worst)