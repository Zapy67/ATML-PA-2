import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from huggingface_hub import HfApi, upload_file
import os

# ⚙️ Config
repo_id = ""  # <-- change to your repo
dataset = ""
output_file = f"{dataset}_resnet50_features.pt"

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

# 6️⃣ Dataset setup
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

torch.save({'features': features, 'labels': labels}, output_file)
print(f"✅ Saved features locally to {output_file}")

# 8️⃣ Upload to Hugging Face Hub 🚀
print(f"⬆️ Uploading {output_file} to {repo_id} ...")

upload_file(
    path_or_fileobj=output_file,
    path_in_repo=output_file,
    repo_id=repo_id,
    repo_type="dataset"
)

print("✅ Upload complete! File is available on Hugging Face Hub.")
