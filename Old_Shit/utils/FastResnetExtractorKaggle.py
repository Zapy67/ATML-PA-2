# Note GPT Generated, and incomplete

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader

### DataParallel

# 1️⃣ Load pretrained ResNet-50 and remove final layer
resnet = models.resnet50(weights="IMAGENET1K_V1")
feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

# 2️⃣ Freeze parameters
for p in feature_extractor.parameters():
    p.requires_grad = False

# 3️⃣ Move to device(s)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = feature_extractor.to(device)

# 4️⃣ Wrap in DataParallel if multiple GPUs are available
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs 🚀")
    feature_extractor = nn.DataParallel(feature_extractor)

# 5️⃣ Compile for optimized graph execution
feature_extractor = torch.compile(feature_extractor, mode="reduce-overhead")

# 6️⃣ Dataset setup (same as before)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])
dataset = datasets.ImageFolder("PACs/photo", transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=8, pin_memory=True)

# 7️⃣ Extract features in parallel
features, labels = [], []
feature_extractor.eval()

with torch.inference_mode():
    for imgs, lbls in loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = feature_extractor(imgs)
        feats = feats.view(feats.size(0), -1)
        features.append(feats.cpu())
        labels.append(lbls)

features = torch.cat(features)
labels = torch.cat(labels)
torch.save({'features': features, 'labels': labels}, 'pacs_photo_features.pt')

print("✅ Feature extraction complete.")


### DDP (DistributedDataParallel) (Much more complicated)

# import torch.distributed as dist
# from torch.utils.data import DistributedSampler
# import os

# def setup(rank, world_size):
#     dist.init_process_group(
#         backend='nccl',
#         init_method='env://',
#         world_size=world_size,
#         rank=rank
#     )
#     torch.cuda.set_device(rank)

# def cleanup():
#     dist.destroy_process_group()

# def main(rank, world_size=2):
#     setup(rank, world_size)

#     # Model setup
#     resnet = models.resnet50(weights="IMAGENET1K_V1")
#     feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
#     feature_extractor.to(rank)
#     feature_extractor = nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[rank])

#     # Dataset
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225],
#         ),
#     ])
#     dataset = datasets.ImageFolder("PACs/photo", transform=transform)
#     sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
#     loader = DataLoader(dataset, batch_size=64, sampler=sampler, num_workers=4, pin_memory=True)

#     # Extraction loop
#     feature_extractor.eval()
#     features, labels = [], []
#     with torch.inference_mode():
#         for imgs, lbls in loader:
#             imgs = imgs.to(rank, non_blocking=True)
#             feats = feature_extractor(imgs).view(imgs.size(0), -1)
#             features.append(feats.cpu())
#             labels.append(lbls)

#     # Save per-rank results
#     torch.save({
#         "features": torch.cat(features),
#         "labels": torch.cat(labels)
#     }, f"pacs_photo_features_rank{rank}.pt")

#     cleanup()

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size)
