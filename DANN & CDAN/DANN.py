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
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt



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
    def __init__(self, pretrained: bool = True, freeze: bool = False):
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
        freeze_backbone: bool = False,
        class_head_dims: list = None,
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
        self.class_head = ClassificationHead(
            self.feature_extractor.output_dim,
            num_classes,
            class_head_dims
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
        class_output = self.class_head(features)
        
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
    
    @staticmethod
    def _unpack_batch(batch):
        """
        Accepts batches that can be (imgs, labels) or (imgs, labels, domains).
        Returns imgs, labels, domains_or_none
        """
        if len(batch) == 2:
            imgs, labels = batch
            domains = None
        elif len(batch) == 3:
            imgs, labels, domains = batch
        else:
            # handle nested tuples: sometimes DataLoader returns ((imgs, labels), domains) etc.
            # try flattening heuristically
            try:
                imgs, labels = batch[0], batch[1]
                domains = batch[2] if len(batch) > 2 else None
            except Exception:
                raise ValueError("Unsupported batch format")
        return imgs, labels, domains
    
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
        
        # move
        source_data = source_data.to(self.device)
        source_labels = source_labels.to(self.device)
        target_data = target_data.to(self.device)

        # domain labels for domain classifier: 0 = source, 1 = target
        domain_labels_source = torch.zeros(source_data.size(0), dtype=torch.long, device=self.device)
        domain_labels_target = torch.ones(target_data.size(0), dtype=torch.long, device=self.device)

        # forward
        class_out_s, domain_out_s, _ = self.model(source_data, alpha)
        _, domain_out_t, _ = self.model(target_data, alpha)

        # losses
        class_loss = self.class_criterion(class_out_s, source_labels)
        
        dom_loss_s = self.domain_criterion(domain_out_s, domain_labels_source)
        dom_loss_t = self.domain_criterion(domain_out_t, domain_labels_target)
        domain_loss = dom_loss_s + dom_loss_t

        total_loss = class_loss + domain_loss

        total_loss.backward()
        self.optimizer.step()
        
        # compute accuracies
        with torch.no_grad():
            class_pred = torch.argmax(class_out_s, dim=1)
            class_acc = (class_pred == source_labels).float().mean().item()

            dom_pred_s = torch.argmax(domain_out_s, dim=1)
            dom_pred_t = torch.argmax(domain_out_t, dim=1)
            dom_acc_s = (dom_pred_s == domain_labels_source).float().mean().item()
            dom_acc_t = (dom_pred_t == domain_labels_target).float().mean().item()
            dom_acc = 0.5 * (dom_acc_s + dom_acc_t)

        return {
            'class_loss': class_loss.item(),
            'domain_loss': domain_loss.item(),
            'total_loss': total_loss.item(),
            'class_acc': class_acc,
            'domain_acc': dom_acc
        }
    
    def evaluate_on_target(self, target_loader: torch.utils.data.DataLoader, return_features: bool = False):
        """
        Evaluate classifier on target_loader (no gradient updates).
        If return_features=True, also return (features, labels) arrays for downstream analysis.
        """
        self.model.eval()
        y_true = []
        y_pred = []
        losses = []
        features_list = []

        with torch.no_grad():
            pbar = tqdm(target_loader, desc="Evaluating target", leave=False)
            for batch in pbar:
                imgs, labels, _ = self._unpack_batch(batch)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # forward: set alpha = 0 for evaluation (no GRL effect)
                try:
                    class_logits, _, feats = self.model(imgs, 0.0)
                except Exception:
                    # fallback: maybe model(imgs) returns class logits and features differently
                    out = self.model(imgs)
                    # try to guess
                    if isinstance(out, tuple) and len(out) >= 1:
                        class_logits = out[0]
                        feats = out[-1] if len(out) > 1 else None
                    else:
                        class_logits = out
                        feats = None

                loss = self.class_criterion(class_logits, labels)
                losses.append(loss.item())

                preds = torch.argmax(class_logits, dim=1)
                y_true.append(labels.cpu())
                y_pred.append(preds.cpu())

                if return_features and feats is not None:
                    features_list.append(feats.detach().cpu())

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else None
        acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else None

        features_np = None
        if return_features and len(features_list) > 0:
            features_np = torch.cat(features_list).numpy()

        return {
            'loss': avg_loss,
            'accuracy': acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'features': features_np
        }


    def train(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        num_epochs: int,
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
            
            # make an iterator for target to pair batches (cycling)
            target_iter = iter(target_loader)
            
            # Training loop
            pbar = tqdm(source_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else source_loader
            
            for source_data, source_labels in pbar:
                # get matching target batch (cycle if needed)
                try:
                    t_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    t_batch = next(target_iter)
                target_imgs, _, _ = self._unpack_batch(t_batch)
                
                # match sizes
                min_b = min(source_imgs.size(0), target_imgs.size(0))
                source_imgs = source_imgs[:min_b]
                source_labels = source_labels[:min_b]
                target_imgs = target_imgs[:min_b]

                metrics = self.train_step(source_imgs, source_labels, target_imgs, alpha)

                # accumulate
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)

                if verbose:
                    pbar.set_postfix({
                        'cls_loss': f"{metrics['class_loss']:.4f}",
                        'dom_loss': f"{metrics['domain_loss']:.4f}",
                        'cls_acc': f"{metrics['class_acc']:.4f}",
                        'alpha': f"{alpha:.4f}"
                    })

            # epoch averages
            for k in epoch_metrics:
                avg_val = float(np.mean(epoch_metrics[k])) if len(epoch_metrics[k]) > 0 else None
                self.history[f'train_{k}'].append(avg_val)

            # evaluate on target dataset (no training)
            tgt_metrics = self.evaluate_on_target(target_loader, return_features=False)
            self.history['target_class_loss'].append(tgt_metrics['loss'])
            self.history['target_class_acc'].append(tgt_metrics['accuracy'])

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {self.history['train_total_loss'][-1]:.4f}, "
                      f"Train Acc: {self.history['train_class_acc'][-1]:.4f}, "
                      f"Target Loss: {tgt_metrics['loss']:.4f}, "
                      f"Target Acc: {tgt_metrics['accuracy']:.4f}")

    def analysis(
        self,
        loader: torch.utils.data.DataLoader,
        class_names: Optional[list] = None,
        tsne_perplexity: int = 30,
        tsne_samples: Optional[int] = 2000,
        pca_before_tsne: bool = True,
        random_state: int = 0,
        color_by: str = 'class'  # 'class' or 'domain'
    ):
        """
        Detailed analysis:
        - computes predictions on the provided loader
        - prints accuracy, classification report
        - plots confusion matrix
        - extracts features and runs t-SNE (optionally pre-reduced by PCA) and plots clusters
        color_by: whether to color t-SNE by 'class' or 'domain' (loader must provide domain if chosen)
        """

        # 1) Get preds, true labels, features, and domains (if present)
        self.model.eval()
        y_true = []
        y_pred = []
        feats_list = []
        domains = []

        with torch.no_grad():
            pbar = tqdm(loader, desc="Collecting features", leave=False)
            for batch in pbar:
                imgs, labels, doms = self._unpack_batch(batch)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                # forward
                try:
                    class_logits, _, feats = self.model(imgs, 0.0)
                except Exception:
                    out = self.model(imgs)
                    if isinstance(out, tuple):
                        class_logits = out[0]
                        feats = out[-1] if len(out) > 1 else None
                    else:
                        class_logits = out
                        feats = None

                preds = torch.argmax(class_logits, dim=1)

                y_true.append(labels.cpu())
                y_pred.append(preds.cpu())
                if feats is not None:
                    feats_list.append(feats.detach().cpu())

                # collect domains if available
                if doms is not None:
                    domains.append(doms.cpu())
                else:
                    domains.append(torch.full_like(labels.cpu(), -1))

        if len(y_true) == 0:
            print("No data collected from loader.")
            return

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        all_domains = torch.cat(domains).numpy() if len(domains) > 0 else None
        features_np = np.concatenate([f.numpy() if isinstance(f, torch.Tensor) else f for f in feats_list], axis=0) if len(feats_list) > 0 else None

        # 2) Print metrics
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {acc*100:.2f}%")
        if class_names is None:
            print(classification_report(y_true, y_pred, digits=4))
        else:
            print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        # 3) Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, cmap="Blues", fmt="d")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.show()

        # 4) t-SNE / PCA visualization (if features available)
        if features_np is not None:
            # sub-sample for speed if requested
            n_total = features_np.shape[0]
            if tsne_samples is None or tsne_samples >= n_total:
                idxs = np.arange(n_total)
            else:
                rng = np.random.RandomState(random_state)
                idxs = rng.choice(n_total, tsne_samples, replace=False)
            feats_sub = features_np[idxs]
            labels_sub = y_true[idxs]
            domains_sub = all_domains[idxs] if all_domains is not None else None

            # optionally PCA reduce to 50 dims before TSNE
            if pca_before_tsne and feats_sub.shape[1] > 50:
                pca = PCA(n_components=50, random_state=random_state)
                feats_sub = pca.fit_transform(feats_sub)

            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init='pca', random_state=random_state)
            tsne_proj = tsne.fit_transform(feats_sub)

            plt.figure(figsize=(10,8))
            if color_by == 'class':
                sc = plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=labels_sub, cmap='tab20', s=6)
                plt.title("t-SNE of features (colored by class)")
                plt.colorbar(sc, label='class id')
            else:
                # color by domain
                if domains_sub is None:
                    print("No domain information available for coloring by domain.")
                else:
                    sc = plt.scatter(tsne_proj[:,0], tsne_proj[:,1], c=domains_sub, cmap='tab10', s=6)
                    plt.title("t-SNE of features (colored by domain)")
                    plt.colorbar(sc, label='domain id')
            plt.show()

        else:
            print("No feature vectors available from model to run t-SNE. Ensure your model returns features as 3rd output.")


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