import torch
from torchvision.models import resnet50, ResNet50_Weights
from torch import nn
from utils import unfreeze_layers
from torch.autograd import grad


import torch
import torch.nn as nn
from torch.autograd import grad

class IRMPenalty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, losses: torch.Tensor, dummy_w: torch.Tensor) -> torch.Tensor:
        grad_even = grad(losses[0::2].mean(), dummy_w, create_graph=True)[0]
        grad_odd  = grad(losses[1::2].mean(), dummy_w, create_graph=True)[0]
        return (grad_even * grad_odd).sum()


class IRMLoss(nn.Module):
    def __init__(self, phi):
        super().__init__()
        self.supervised_loss = nn.CrossEntropyLoss(reduction='none')
        self.irm_penalty = IRMPenalty()
        self.phi = phi

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, dummy_w: torch.Tensor):
        weighted_outputs = outputs * dummy_w
        losses = self.supervised_loss(weighted_outputs, targets)
        penalty = self.irm_penalty(losses, dummy_w)
        error = losses.mean()
        return error, self.phi*penalty

def resnet_classifier(num_classes, device='cpu'):
    classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, num_classes)
    classifier.to(device)
    return classifier