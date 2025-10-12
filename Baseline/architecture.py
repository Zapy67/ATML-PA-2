import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

def resnet_classifier(num_classes, device='cpu'):
    classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, num_classes)
    classifier.to(device)
    return classifier

    
