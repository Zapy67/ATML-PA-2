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

from utils.architecture import GradientReversalLayer, ClassificationHead, ResNet50FeatureExtractor


# ============================================================================
# Multilinear Conditioning
# ============================================================================

class MultilinearMap(nn.Module):
    """
    Multilinear map for conditioning domain discriminator on classifier predictions.
    Implements the multilinear conditioning: h ⊗ g where:
    - h: feature representation
    - g: classifier predictions (softmax probabilities)
    - ⊗: Matrix Product
    
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
    def __init__(self, input_dim: int, hidden_dims: list = [1024, 512, 128]):
        super(ConditionalDomainDiscriminator, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
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
        
        return weights/weights.mean()


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
        resnet: nn.Module,
        class_head_dims: list = None,
        multilinear_output_dim: int = 1024,
        domain_discriminator_dims: list = [1024, 512, 128],
        use_entropy: bool = False
    ):
        super(CDAN, self).__init__()
        
        # ResNet-50 feature extractor
        self.feature_extractor = resnet
        
        # Label predictor
        self.class_head = ClassificationHead(
            self.feature_extractor.output_dim,
            num_classes,
            class_head_dims
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
        class_output = self.class_head(features)
        
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
        return self.class_head(features)
    

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
        weight_decay: float = 1e-4,
        gamma: float = 10.0,
        max_grad_norm: Optional[float] = None,
        label_smoothing = 0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.label_smoothing = label_smoothing
        
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters(), 'lr': learning_rate * 0.5},
            {'params': model.class_head.parameters(), 'lr': learning_rate},
            {'params': model.multilinear_map.parameters(), 'lr': learning_rate},
            {'params': model.domain_discriminator.parameters(), 'lr': learning_rate},
        ], weight_decay=weight_decay)

        
        # Loss functions
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss(reduction='none' if model.use_entropy else 'mean')  # No reduction for entropy weighting
        
        # Training history
        self.history = {
            'train_class_loss': [],
            'train_domain_loss': [],
            'train_total_loss': [],
            'train_class_acc': [],
            'train_domain_acc': [],
            'target_class_loss': [],
            'target_class_acc': []
        }

    def _smooth_domain_labels(self, domain_labels: torch.Tensor) -> torch.Tensor:
        """
        Apply label smoothing to domain labels.
        
        Args:
            domain_labels: Hard labels (0 for source, 1 for target)
        
        Returns:
            Smoothed labels as one-hot vectors with smoothing applied
        """
        if self.label_smoothing == 0.0:
            return domain_labels
        
        # Create one-hot encoding
        num_classes = 2  # binary domain classification
        batch_size = domain_labels.size(0)
        
        # Initialize with smoothing value
        smooth_labels = torch.full(
            (batch_size, num_classes),
            self.label_smoothing / num_classes,
            device=self.device
        )
        
        # Set target class to (1 - smoothing + smoothing/num_classes)
        smooth_labels.scatter_(
            1, 
            domain_labels.unsqueeze(1), 
            1.0 - self.label_smoothing + self.label_smoothing / num_classes
        )
        
        return smooth_labels

    def _compute_smooth_domain_loss(
        self, 
        domain_output: torch.Tensor, 
        domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain loss with label smoothing.
        
        Args:
            domain_output: Domain discriminator logits (batch_size, 2)
            domain_labels: Hard domain labels (batch_size,)
        
        Returns:
            Loss value (scalar or per-sample for entropy weighting)
        """
        if self.label_smoothing == 0.0:
            # No smoothing: use standard CrossEntropyLoss
            return self.domain_criterion(domain_output, domain_labels)
        
        # Apply label smoothing
        smooth_labels = self._smooth_domain_labels(domain_labels)
        
        # Compute log probabilities
        log_probs = torch.nn.functional.log_softmax(domain_output, dim=1)
        
        # Compute loss: -sum(smooth_labels * log_probs)
        loss = -torch.sum(smooth_labels * log_probs, dim=1)
        
        # Return mean or per-sample loss depending on entropy weighting
        if self.model.use_entropy:
            return loss  # per-sample for entropy weighting
        else:
            return loss.mean()


    @staticmethod
    def _unpack_batch(batch):
        imgs, labels = batch[:2]
        return imgs, labels, None
    
    def train_step(self, s_imgs, s_labels, t_imgs, alpha):
        self.model.train()
        self.optimizer.zero_grad()

        s_imgs, s_labels, t_imgs = s_imgs.to(self.device), s_labels.to(self.device), t_imgs.to(self.device)

        domain_src = torch.zeros(s_imgs.size(0), dtype=torch.long, device=self.device)
        domain_tgt = torch.ones(t_imgs.size(0), dtype=torch.long, device=self.device)

        # Forward
        s_class, s_domain, _ = self.model(s_imgs, alpha)
        t_class, t_domain, _ = self.model(t_imgs, alpha)

        # Losses
        class_loss = self.class_criterion(s_class, s_labels)
        d_loss_src = self._compute_smooth_domain_loss(s_domain, domain_src)
        d_loss_tgt = self._compute_smooth_domain_loss(t_domain, domain_tgt)
        # d_loss_src = self.domain_criterion(s_domain, domain_src)
        # d_loss_tgt = self.domain_criterion(t_domain, domain_tgt)

        if self.model.use_entropy:
            weights_src = self.model.entropy_weighting(s_class)
            weights_tgt = self.model.entropy_weighting(t_class)
            d_loss = torch.mean(weights_src * d_loss_src) + torch.mean(weights_tgt * d_loss_tgt)
        else:
            d_loss = d_loss_src + d_loss_tgt

        total_loss = class_loss + 2.0*d_loss
        total_loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        with torch.no_grad():
            cls_acc = (torch.argmax(s_class, 1) == s_labels).float().mean().item()
            dom_acc_s = (torch.argmax(s_domain, 1) == domain_src).float().mean().item()
            dom_acc_t = (torch.argmax(t_domain, 1) == domain_tgt).float().mean().item()

        return {
            'class_loss': class_loss.item(),
            'domain_loss': d_loss.item(),
            'total_loss': total_loss.item(),
            'class_acc': cls_acc,
            'domain_acc': 0.5 * (dom_acc_s + dom_acc_t)
        }

    def evaluate_on_target(self, loader, return_features=False):
        self.model.eval()
        y_true, y_pred, losses, feats = [], [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating target", leave=False):
                imgs, labels, _ = self._unpack_batch(batch)
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                class_logits, _, f = self.model(imgs, 0.0)
                loss = self.class_criterion(class_logits, labels)
                preds = torch.argmax(class_logits, 1)

                y_true.append(labels.cpu())
                y_pred.append(preds.cpu())
                losses.append(loss.item())
                if return_features:
                    feats.append(f.cpu())

        y_true, y_pred = torch.cat(y_true).numpy(), torch.cat(y_pred).numpy()
        acc = float((y_true == y_pred).mean())
        result = {'loss': float(np.mean(losses)), 'accuracy': acc}
        if return_features:
            result['features'] = torch.cat(feats).numpy()
        return result

    def train(self, source_loader, target_loader, num_epochs, verbose=True):
        for epoch in range(num_epochs):
            alpha = compute_lambda_schedule(epoch, num_epochs, self.gamma)
            metrics_epoch = {k: [] for k in ['class_loss', 'domain_loss', 'total_loss', 'class_acc', 'domain_acc']}

            target_iter = iter(target_loader)
            pbar = tqdm(source_loader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else source_loader

            for batch in pbar:
                s_imgs, s_labels, _ = self._unpack_batch(batch)
                try:
                    t_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    t_batch = next(target_iter)
                t_imgs, _, _ = self._unpack_batch(t_batch)

                min_b = min(s_imgs.size(0), t_imgs.size(0))
                step_metrics = self.train_step(s_imgs[:min_b], s_labels[:min_b], t_imgs[:min_b], alpha)

                for k, v in step_metrics.items():
                    metrics_epoch[k].append(v)

                if verbose:
                    pbar.set_postfix({
                        'cls_loss': f"{step_metrics['class_loss']:.4f}",
                        'dom_loss': f"{step_metrics['domain_loss']:.4f}",
                        'cls_acc': f"{step_metrics['class_acc']:.4f}",
                        'alpha': f"{alpha:.4f}"
                    })
                
            for k in metrics_epoch:
                self.history[f"train_{k}"].append(np.mean(metrics_epoch[k]))

            tgt = self.evaluate_on_target(target_loader)
            self.history['target_class_loss'].append(tgt['loss'])
            self.history['target_class_acc'].append(tgt['accuracy'])

            if verbose:
                print(f"Epoch {epoch+1}/{num_epochs} - "
                      f"Train Loss: {self.history['train_total_loss'][-1]:.4f}, "
                      f"Train Acc: {self.history['train_class_acc'][-1]:.4f}, "
                      f"Target Acc: {tgt['accuracy']:.4f}")

    def analysis(
        self,
        src_loader,
        tgt_loader,
        class_names=None,
        tsne_perplexity=30,
        tsne_samples=2000,
        pca_before_tsne=True,
        random_state=0,
        normalize_cm=True,
    ):
        """
        Evaluate model on source vs target loaders.
        Shows classification reports, confusion matrices, and t-SNE feature plots.

        This works for CDAN (or DANN) models that return either:
        (class_logits, domain_logits, features)
        or just logits/features in inference.
        """
        import torch
        import numpy as np
        import seaborn as sns
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from tqdm import tqdm

        self.model.eval()
        results = {}

        def eval_loader(name, loader):
            y_true, y_pred, feats_list = [], [], []

            with torch.no_grad():
                pbar = tqdm(loader, desc=f"Evaluating {name}", leave=False)
                for batch in pbar:
                    # accept (imgs, labels) or (imgs, labels, domain)
                    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
                        imgs, labels = batch[0], batch[1]
                    else:
                        raise ValueError("Loader must yield (imgs, labels[, ...]) tuples")

                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)

                    out = self.model(imgs, 0.0)  # alpha=0 disables GRL for eval

                    if isinstance(out, tuple):
                        cls_logits = out[0]
                        feats = out[2] if len(out) > 2 else None
                    else:
                        # fallback: model returns logits only
                        cls_logits = out
                        feats = None

                    preds = torch.argmax(cls_logits, dim=1)
                    y_true.append(labels.cpu())
                    y_pred.append(preds.cpu())

                    if feats is not None:
                        feats_list.append(feats.cpu())

            if len(y_true) == 0:
                return {"report": None, "cm": None, "features": None, "y_true": np.array([])}

            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
            feats = torch.cat(feats_list).numpy() if len(feats_list) > 0 else None

            # --- Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=class_names if class_names else None,
                digits=4
            )
            print(f"\n=== {name.upper()} REPORT ===\n{report}")

            # --- Confusion matrix
            labels_range = np.arange(0, max(int(y_true.max()), int(y_pred.max())) + 1)
            cm = confusion_matrix(y_true, y_pred, labels=labels_range)

            if normalize_cm:
                cm_disp = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            else:
                cm_disp = cm

            # Large heatmap without annotations for clarity
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_disp,
                annot=False,
                cmap="Blues",
                xticklabels=False,
                yticklabels=False,
            )
            plt.title(f"Confusion Matrix ({name})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()

            # Top-10 most frequent true classes (so the heatmap is readable)
            conf_sum = np.sum(cm, axis=1)
            topk = min(10, conf_sum.size)
            topk_idx = np.argsort(-conf_sum)[:topk]
            if topk > 0:
                cm_topk = cm_disp[topk_idx][:, topk_idx]
                labels_topk = [class_names[i] if class_names is not None else str(i) for i in topk_idx]

                plt.figure(figsize=(10, 8))
                sns.heatmap(cm_topk, annot=True, fmt=".2f", cmap="Blues",
                            xticklabels=labels_topk, yticklabels=labels_topk)
                plt.title(f"Top-{topk} Confusion Matrix ({name})")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.tight_layout()
                plt.show()

            return {"report": report, "cm": cm, "features": feats, "y_true": y_true}

        # --- Evaluate both domains
        src_res = eval_loader("source", src_loader)
        tgt_res = eval_loader("target", tgt_loader)

        # --- t-SNE visualization (only if both sides have features)
        if (src_res["features"] is not None) and (tgt_res["features"] is not None):
            src_feats, tgt_feats = src_res["features"], tgt_res["features"]

            rng = np.random.RandomState(random_state)
            n_src = min(src_feats.shape[0], tsne_samples // 2)
            n_tgt = min(tgt_feats.shape[0], tsne_samples // 2)

            # guard for tiny sets
            if n_src > 0 and n_tgt > 0:
                src_idx = rng.choice(src_feats.shape[0], n_src, replace=False)
                tgt_idx = rng.choice(tgt_feats.shape[0], n_tgt, replace=False)

                X = np.vstack([src_feats[src_idx], tgt_feats[tgt_idx]])
                y_domain = np.array([0] * n_src + [1] * n_tgt)  # 0=src, 1=tgt

                # optional PCA before t-SNE
                if pca_before_tsne and X.shape[1] > 50:
                    X = PCA(n_components=50, random_state=random_state).fit_transform(X)

                emb = TSNE(n_components=2, perplexity=tsne_perplexity, random_state=random_state).fit_transform(X)

                plt.figure(figsize=(8, 6))
                plt.scatter(emb[y_domain == 0, 0], emb[y_domain == 0, 1], s=10, alpha=0.7, label="Source")
                plt.scatter(emb[y_domain == 1, 0], emb[y_domain == 1, 1], s=10, alpha=0.7, label="Target")
                plt.legend()
                plt.title("t-SNE of Source vs Target Features")
                plt.tight_layout()
                plt.show()

                results["tsne"] = emb
            else:
                print("Not enough features for t-SNE (need >=1 sample per domain).")

        results["source"] = src_res
        results["target"] = tgt_res
        return results

