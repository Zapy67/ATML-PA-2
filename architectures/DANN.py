"""
Domain-Adversarial Neural Network (DANN) Implementation
Based on "Domain-Adversarial Training of Neural Networks" by Ganin et al.
Uses ResNet-50 as frozen feature extractor backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple, Optional, Dict
import numpy as np
from tqdm import tqdm


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
    def __init__(self, pretrained: bool = True, freeze: bool = True):
        super(ResNet50FeatureExtractor, self).__init__()
        
        # Load pretrained ResNet-50
        resnet50 = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        # ResNet-50 outputs 2048-dimensional features
        self.features = nn.Sequential(*list(resnet50.children())[:-1])
        self.output_dim = 2048
        
        # Freeze the backbone if specified
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
            self.features.eval()
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        Returns:
            features: Feature tensor of shape (batch_size, 2048)
        """
        with torch.set_grad_enabled(self.training and not self._is_frozen()):
            features = self.features(x)
            # Flatten the output
            features = features.view(features.size(0), -1)
        return features
    
    def _is_frozen(self):
        """Check if the backbone is frozen"""
        return not next(self.features.parameters()).requires_grad
    
    def unfreeze(self):
        """Unfreeze the backbone for fine-tuning"""
        for param in self.features.parameters():
            param.requires_grad = True
        self.features.train()
    
    def freeze(self):
        """Freeze the backbone"""
        for param in self.features.parameters():
            param.requires_grad = False
        self.features.eval()


# ============================================================================
# Label Predictor
# ============================================================================

class LabelPredictor(nn.Module):
    """
    Label predictor network for classification task
    """
    def __init__(self, input_dim: int, num_classes: int, hidden_dims: list = None):
        super(LabelPredictor, self).__init__()
        
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


# ============================================================================
# Domain Discriminator
# ============================================================================

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator network for adversarial training
    Predicts whether features come from source (0) or target (1) domain
    """
    def __init__(self, input_dim: int, hidden_dims: list = None):
        super(DomainDiscriminator, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256]
        
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
        
        layers.append(nn.Linear(prev_dim, 2))  # Binary classification
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.discriminator(x)


# ============================================================================
# DANN Model
# ============================================================================

class DANN(nn.Module):
    """
    Complete DANN model with ResNet-50 backbone, combining feature extractor, 
    label predictor, and domain discriminator with gradient reversal layer.
    """
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        label_predictor_dims: list = None,
        domain_discriminator_dims: list = [1024, 512],
        dropout: float = 0.5
    ):
        super(DANN, self).__init__()
        
        # ResNet-50 feature extractor (frozen by default)
        self.feature_extractor = ResNet50FeatureExtractor(
            pretrained=pretrained,
            freeze=freeze_backbone
        )
        
        # Label predictor
        self.label_predictor = LabelPredictor(
            self.feature_extractor.output_dim,
            num_classes,
            label_predictor_dims
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer()
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            self.feature_extractor.output_dim,
            domain_discriminator_dims
        )
    
    def forward(self, x, alpha: Optional[float] = None):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            alpha: Gradient reversal multiplier (if None, uses current value)
        
        Returns:
            class_output: Class predictions
            domain_output: Domain predictions
            features: Extracted features
        """
        # Extract features using ResNet-50
        features = self.feature_extractor(x)
        
        # Predict class labels
        class_output = self.label_predictor(features)
        
        # Update GRL lambda if provided
        if alpha is not None:
            self.grl.set_lambda(alpha)
        
        # Apply gradient reversal and predict domain
        reversed_features = self.grl(features)
        domain_output = self.domain_discriminator(reversed_features)
        
        return class_output, domain_output, features
    
    def predict(self, x):
        """Predict class labels only (for inference)"""
        features = self.feature_extractor(x)
        return self.label_predictor(features)
    
    def unfreeze_backbone(self):
        """Unfreeze ResNet-50 backbone for fine-tuning"""
        self.feature_extractor.unfreeze()
    
    def freeze_backbone(self):
        """Freeze ResNet-50 backbone"""
        self.feature_extractor.freeze()


# ============================================================================
# Training Utilities
# ============================================================================

def compute_lambda_schedule(epoch: int, max_epochs: int, gamma: float = 10) -> float:
    """
    Compute lambda for gradient reversal layer schedule as per DANN paper
    
    Args:
        epoch: Current epoch
        max_epochs: Total number of epochs
        gamma: Hyperparameter controlling the rate of adaptation
    
    Returns:
        lambda value
    """
    p = float(epoch) / max_epochs
    return 2. / (1. + np.exp(-gamma * p)) - 1.


# ============================================================================
# DANN Trainer
# ============================================================================

class DANNTrainer:
    """
    Trainer class for DANN model
    """
    def __init__(
        self,
        model: DANN,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        gamma: float = 10.0
    ):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        
        # Optimizers
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_class_loss': [],
            'train_domain_loss': [],
            'train_total_loss': [],
            'train_class_acc': [],
            'train_domain_acc': [],
            'val_class_loss': [],
            'val_class_acc': []
        }
    
    def train_step(
        self,
        source_data: torch.Tensor,
        source_labels: torch.Tensor,
        target_data: torch.Tensor,
        alpha: float
    ) -> Dict[str, float]:
        """
        Single training step for DANN
        
        Args:
            source_data: Source domain data
            source_labels: Source domain labels
            target_data: Target domain data
            alpha: Current lambda value for gradient reversal
        
        Returns:
            Dictionary with loss and accuracy metrics
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_size = source_data.size(0)
        
        # Move data to device
        source_data = source_data.to(self.device)
        source_labels = source_labels.to(self.device)
        target_data = target_data.to(self.device)
        
        # Create domain labels (0 for source, 1 for target)
        domain_labels_source = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        domain_labels_target = torch.ones(target_data.size(0), dtype=torch.long).to(self.device)
        
        # Forward pass for source data
        class_output_source, domain_output_source, _ = self.model(source_data, alpha)
        
        # Forward pass for target data
        _, domain_output_target, _ = self.model(target_data, alpha)
        
        # Compute losses
        class_loss = self.class_criterion(class_output_source, source_labels)
        
        domain_loss_source = self.domain_criterion(domain_output_source, domain_labels_source)
        domain_loss_target = self.domain_criterion(domain_output_target, domain_labels_target)
        domain_loss = domain_loss_source + domain_loss_target
        
        total_loss = class_loss + domain_loss
        
        # Backward pass and optimization
        total_loss.backward()
        self.optimizer.step()
        
        # Compute accuracies
        class_pred = torch.argmax(class_output_source, dim=1)
        class_acc = (class_pred == source_labels).float().mean().item()
        
        domain_pred_source = torch.argmax(domain_output_source, dim=1)
        domain_pred_target = torch.argmax(domain_output_target, dim=1)
        domain_acc_source = (domain_pred_source == domain_labels_source).float().mean().item()
        domain_acc_target = (domain_pred_target == domain_labels_target).float().mean().item()
        domain_acc = (domain_acc_source + domain_acc_target) / 2
        
        return {
            'class_loss': class_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item(),
            'class_acc': class_acc,
            'domain_acc': domain_acc
        }
    
    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True
    ):
        """
        Train DANN model
        
        Args:
            source_loader: DataLoader for source domain
            target_loader: DataLoader for target domain
            num_epochs: Number of training epochs
            val_loader: Optional validation DataLoader
            verbose: Whether to print progress
        """
        for epoch in range(num_epochs):
            # Compute lambda schedule
            alpha = compute_lambda_schedule(epoch, num_epochs, self.gamma)
            
            epoch_metrics = {
                'class_loss': [],
                'domain_loss': [],
                'total_loss': [],
                'class_acc': [],
                'domain_acc': []
            }
            
            # Create iterator for target loader
            target_iter = iter(target_loader)
            
            # Training loop
            pbar = tqdm(source_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else source_loader
            
            for source_data, source_labels in pbar:
                # Get target data (cycle if necessary)
                try:
                    target_data, _ = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_data, _ = next(target_iter)
                
                # Match batch sizes
                min_batch = min(source_data.size(0), target_data.size(0))
                source_data = source_data[:min_batch]
                source_labels = source_labels[:min_batch]
                target_data = target_data[:min_batch]
                
                # Training step
                metrics = self.train_step(source_data, source_labels, target_data, alpha)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                
                if verbose:
                    pbar.set_postfix({
                        'cls_loss': f"{metrics['class_loss']:.4f}",
                        'dom_loss': f"{metrics['domain_loss']:.4f}",
                        'cls_acc': f"{metrics['class_acc']:.4f}",
                        'alpha': f"{alpha:.4f}"
                    })
            
            # Compute epoch averages
            for key in epoch_metrics:
                avg_value = np.mean(epoch_metrics[key])
                self.history[f'train_{key}'].append(avg_value)
            
            # Validation
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                self.history['val_class_loss'].append(val_metrics['loss'])
                self.history['val_class_acc'].append(val_metrics['accuracy'])
                
                if verbose:
                    print(f"Epoch {epoch+1}/{num_epochs} - "
                          f"Train Loss: {self.history['train_total_loss'][-1]:.4f}, "
                          f"Train Acc: {self.history['train_class_acc'][-1]:.4f}, "
                          f"Val Loss: {val_metrics['loss']:.4f}, "
                          f"Val Acc: {val_metrics['accuracy']:.4f}")
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on a dataset
        
        Args:
            data_loader: DataLoader for evaluation
        
        Returns:
            Dictionary with loss and accuracy metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # Predict class labels only
                outputs = self.model.predict(data)
                loss = self.class_criterion(outputs, labels)
                
                total_loss += loss.item()
                pred = torch.argmax(outputs, dim=1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': correct / total
        }
    
    def test(self, test_loader: DataLoader, domain_name: str = 'test') -> Dict[str, float]:
        """
        Test model on a dataset
        
        Args:
            test_loader: DataLoader for testing
            domain_name: Name of the domain for logging
        
        Returns:
            Dictionary with test metrics
        """
        metrics = self.evaluate(test_loader)
        print(f"{domain_name.capitalize()} Results:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        return metrics


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example parameters for image classification
    num_classes = 10  # e.g., 10 classes for digit recognition
    batch_size = 32
    num_epochs = 50
    
    # Create model with ResNet-50 backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DANN(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,  # Frozen ResNet-50
        label_predictor_dims=[512, 256],  # Additional layers after ResNet
        domain_discriminator_dims=[1024, 512]
    )
    
    # Create trainer
    trainer = DANNTrainer(
        model=model,
        device=device,
        learning_rate=1e-3,
        weight_decay=5e-4,
        gamma=10.0
    )
    
    print("DANN model with ResNet-50 backbone initialized successfully!")
    print(f"Feature extractor output dim: {model.feature_extractor.output_dim}")
    
    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters (ResNet-50): {frozen_params:,}")
    print(f"Device: {device}")
    
    # Example: Create dummy data to test the model
    dummy_input = torch.randn(4, 3, 224, 224).to(device)  # 4 images, 3 channels, 224x224
    class_out, domain_out, features = model(dummy_input, alpha=0.5)
    print(f"\nTest forward pass:")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Class output shape: {class_out.shape}")
    print(f"Domain output shape: {domain_out.shape}")