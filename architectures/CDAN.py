"""
Conditional Domain Adversarial Network (CDAN) Implementation
Based on "Conditional Adversarial Domain Adaptation" by Long et al. (NeurIPS 2018)
Uses ResNet-50 as frozen feature extractor backbone

Paper: https://arxiv.org/abs/1705.10667
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import models
from typing import Tuple, Optional, Dict, List
import numpy as np
from tqdm import tqdm


# ============================================================================
# Gradient Reversal Layer
# ============================================================================

class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer for adversarial training.
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
# Label Predictor (Classifier)
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
        
        self.num_classes = num_classes
    
    def forward(self, x):
        return self.classifier(x)


# ============================================================================
# Multilinear Conditioning
# ============================================================================

class MultilinearMap(nn.Module):
    """
    Multilinear map for conditioning domain discriminator on classifier predictions.
    Implements the multilinear conditioning: h ⊗ g where:
    - h: feature representation
    - g: classifier predictions (softmax probabilities)
    
    This is the key component that makes CDAN "conditional" on predicted labels.
    """
    def __init__(self, feature_dim: int, num_classes: int, output_dim: int = None):
        super(MultilinearMap, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Output dimension for the multilinear map
        if output_dim is None:
            output_dim = feature_dim
        
        self.output_dim = output_dim
        
        # Random multilinear map (can also be learned)
        # Paper uses random projection for efficiency
        self.map = nn.Linear(feature_dim * num_classes, output_dim, bias=False)
        
        # Initialize with small random values
        nn.init.normal_(self.map.weight, mean=0, std=0.01)
    
    def forward(self, features: torch.Tensor, predictions: torch.Tensor):
        """
        Args:
            features: Feature tensor of shape (batch_size, feature_dim)
            predictions: Classifier predictions (logits) of shape (batch_size, num_classes)
        
        Returns:
            Multilinear conditioning output of shape (batch_size, output_dim)
        """
        # Get softmax probabilities
        softmax_predictions = F.softmax(predictions, dim=1)
        
        # Outer product: features ⊗ predictions
        # (batch_size, feature_dim, 1) × (batch_size, 1, num_classes)
        # = (batch_size, feature_dim, num_classes)
        outer_product = torch.bmm(
            features.unsqueeze(2),
            softmax_predictions.unsqueeze(1)
        )
        
        # Flatten to (batch_size, feature_dim * num_classes)
        outer_product = outer_product.view(features.size(0), -1)
        
        # Apply random projection
        return self.map(outer_product)


# ============================================================================
# Conditional Domain Discriminator
# ============================================================================

class ConditionalDomainDiscriminator(nn.Module):
    """
    Conditional domain discriminator that takes multilinear conditioned features.
    Predicts whether features come from source (0) or target (1) domain,
    conditioned on the classifier's predictions.
    """
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 1024]):
        super(ConditionalDomainDiscriminator, self).__init__()
        
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
        
        # Binary classification (source vs target)
        layers.append(nn.Linear(prev_dim, 2))
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.discriminator(x)


# ============================================================================
# Entropy-based Weight (for CDAN+E variant)
# ============================================================================

class EntropyWeighting(nn.Module):
    """
    Entropy-based weighting for CDAN+E variant.
    Weights samples by their prediction entropy to focus on easy-to-transfer samples.
    
    Weight = 1 + exp(-entropy)
    """
    def __init__(self):
        super(EntropyWeighting, self).__init__()
    
    def forward(self, predictions: torch.Tensor):
        """
        Args:
            predictions: Classifier logits of shape (batch_size, num_classes)
        
        Returns:
            weights: Sample weights of shape (batch_size,)
        """
        # Compute softmax probabilities
        softmax = F.softmax(predictions, dim=1)
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(softmax * torch.log(softmax + 1e-10), dim=1)
        
        # Compute weights: 1 + exp(-entropy)
        weights = 1.0 + torch.exp(-entropy)
        
        return weights


# ============================================================================
# CDAN Model
# ============================================================================

class CDAN(nn.Module):
    """
    Conditional Domain Adversarial Network (CDAN) with ResNet-50 backbone.
    
    Key difference from DANN: Domain discriminator is conditioned on classifier
    predictions via multilinear map, enabling class-aware alignment.
    """
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        label_predictor_dims: list = None,
        multilinear_output_dim: int = 1024,
        domain_discriminator_dims: list = [1024, 1024],
        use_entropy: bool = False
    ):
        super(CDAN, self).__init__()
        
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
        
        # Multilinear map for conditioning
        self.multilinear_map = MultilinearMap(
            feature_dim=self.feature_extractor.output_dim,
            num_classes=num_classes,
            output_dim=multilinear_output_dim
        )
        
        # Gradient reversal layer
        self.grl = GradientReversalLayer()
        
        # Conditional domain discriminator
        self.domain_discriminator = ConditionalDomainDiscriminator(
            input_dim=multilinear_output_dim,
            hidden_dims=domain_discriminator_dims
        )
        
        # Entropy weighting (for CDAN+E variant)
        self.use_entropy = use_entropy
        if use_entropy:
            self.entropy_weighting = EntropyWeighting()
        
        self.num_classes = num_classes
    
    def forward(self, x, alpha: Optional[float] = None, return_weights: bool = False):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
            alpha: Gradient reversal multiplier
            return_weights: Whether to return entropy weights (for CDAN+E)
        
        Returns:
            class_output: Class predictions
            domain_output: Domain predictions
            features: Extracted features
            weights: Entropy weights (only if use_entropy=True and return_weights=True)
        """
        # Extract features using ResNet-50
        features = self.feature_extractor(x)
        
        # Predict class labels
        class_output = self.label_predictor(features)
        
        # Multilinear conditioning: combine features and predictions
        conditioned_features = self.multilinear_map(features, class_output)
        
        # Update GRL lambda if provided
        if alpha is not None:
            self.grl.set_lambda(alpha)
        
        # Apply gradient reversal
        reversed_features = self.grl(conditioned_features)
        
        # Predict domain with conditioned features
        domain_output = self.domain_discriminator(reversed_features)
        
        # Compute entropy weights if needed
        weights = None
        if self.use_entropy and return_weights:
            weights = self.entropy_weighting(class_output)
        
        if weights is not None:
            return class_output, domain_output, features, weights
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
    Compute lambda for gradient reversal layer schedule
    
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
# CDAN Trainer
# ============================================================================

class CDANTrainer:
    """
    Trainer class for CDAN model
    """
    def __init__(
        self,
        model: CDAN,
        device: torch.device,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
        gamma: float = 10.0
    ):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss(reduction='none')  # No reduction for entropy weighting
        
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
        Single training step for CDAN
        
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
        if self.model.use_entropy:
            class_output_source, domain_output_source, _, weights_source = \
                self.model(source_data, alpha, return_weights=True)
        else:
            class_output_source, domain_output_source, _ = self.model(source_data, alpha)
            weights_source = None
        
        # Forward pass for target data
        if self.model.use_entropy:
            _, domain_output_target, _, weights_target = \
                self.model(target_data, alpha, return_weights=True)
        else:
            _, domain_output_target, _ = self.model(target_data, alpha)
            weights_target = None
        
        # Compute classification loss (only on source)
        class_loss = self.class_criterion(class_output_source, source_labels)
        
        # Compute domain loss with optional entropy weighting
        domain_loss_source = self.domain_criterion(domain_output_source, domain_labels_source)
        domain_loss_target = self.domain_criterion(domain_output_target, domain_labels_target)
        
        if self.model.use_entropy:
            # Apply entropy weights (CDAN+E)
            domain_loss_source = (domain_loss_source * weights_source).mean()
            domain_loss_target = (domain_loss_target * weights_target).mean()
        else:
            domain_loss_source = domain_loss_source.mean()
            domain_loss_target = domain_loss_target.mean()
        
        domain_loss = domain_loss_source + domain_loss_target
        
        # Total loss
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
        Train CDAN model
        
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
    num_classes = 31  # e.g., Office-31 dataset
    batch_size = 32
    num_epochs = 50
    
    # Create CDAN model with ResNet-50 backbone
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # CDAN (without entropy weighting)
    model_cdan = CDAN(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        label_predictor_dims=[512, 256],
        multilinear_output_dim=1024,
        domain_discriminator_dims=[1024, 1024],
        use_entropy=False  # Set to True for CDAN+E variant
    )
    
    # Create trainer
    trainer = CDANTrainer(
        model=model_cdan,
        device=device,
        learning_rate=1e-3,
        weight_decay=5e-4,
        gamma=10.0
    )
    
    print("=" * 80)
    print("CDAN Model with ResNet-50 Backbone")
    print("=" * 80)
    print(f"Feature extractor output dim: {model_cdan.feature_extractor.output_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Multilinear output dim: {model_cdan.multilinear_map.output_dim}")
    print(f"Entropy weighting: {model_cdan.use_entropy}")
    
    # Count trainable vs frozen parameters
    trainable_params = sum(p.numel() for p in model_cdan.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_cdan.parameters())
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Counts:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters (ResNet-50): {frozen_params:,}")
    print(f"  Device: {device}")
    
    # Example: Create dummy data to test the model
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)
    dummy_input = torch.randn(4, 3, 224, 224).to(device)
    
    # Test CDAN
    class_out, domain_out, features = model_cdan(dummy_input, alpha=0.5)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Features shape: {features.shape}")
    print(f"Class output shape: {class_out.shape}")
    print(f"Domain output shape: {domain_out.shape}")
    
    # Test CDAN+E
    print("\n" + "=" * 80)
    print("CDAN+E Variant (with Entropy Weighting)")
    print("=" * 80)
    model_cdan_e = CDAN(
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=True,
        use_entropy=True
    )
    model_cdan_e = model_cdan_e.to(device)
    
    class_out, domain_out, features, weights = model_cdan_e(
        dummy_input, alpha=0.5, return_weights=True
    )
    print(f"Entropy weights shape: {weights.shape}")
    print(f"Entropy weights (sample): {weights[:4]}")