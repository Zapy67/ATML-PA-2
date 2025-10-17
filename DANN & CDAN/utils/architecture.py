import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torchvision import models
from typing import Tuple, Optional, Dict, List


# ============================================================================
# Gradient Reversal Layer
# ============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from the DANN paper.
    Forward pass: identity transformation
    Backward pass: multiplies gradient by -lambda
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal"""
    def __init__(self, lambda_=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
    
    def set_lambda(self, lambda_):
        """Update lambda parameter during training"""
        self.lambda_ = lambda_


# ============================================================================
# Feature Extractor (ResNet-50 Backbone)
# ============================================================================

class ResNet50FeatureExtractor(nn.Module):
    """
    Feature extractor using frozen ResNet-50 backbone
    Extracts features from the last layer before classification
    """
    def __init__(self, pretrained: bool = True):
        super(ResNet50FeatureExtractor, self).__init__()
        
        # Load pretrained ResNet-50
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        # ResNet-50 outputs 2048-dimensional features
        self.backbone = nn.Sequential(*list(resnet50.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output_dim = 2048
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        Returns:
            features: Feature tensor of shape (batch_size, 2048)
        """
        feat_map = self.backbone(x) # (B, C, H, W)
        pooled = self.global_pool(feat_map) # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1) # (B, C)
        return pooled



# ============================================================================
# Classification Head
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Classification Head for classification task
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = None):
        super(ClassificationHead, self).__init__()
        
        if hidden_dims is None:
            # Simple classifier
            self.classifier = nn.Linear(input_dim, num_classes)
        else:
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.classifier(x)