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
# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     accuracy_score
# )
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# import seaborn as sns
# import matplotlib.pyplot as plt

from utils.architecture import GradientReversalLayer, ClassificationHead, ResNet50FeatureExtractor


# ============================================================================
# Domain Discriminator
# ============================================================================

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator network for adversarial training
    Predicts whether features come from source (0) or target (1) domain
    """
    def __init__(self, input_dim: int, hidden_dims: list = None, num_domains: int = 2):
        super(DomainDiscriminator, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [256]
        
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
        
        layers.append(nn.Linear(prev_dim, num_domains))  # classification
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
        resnet: nn.Module,
        pretrained: bool = True,
        class_head_dims: list = None,
        domain_discriminator_dims: list = [1024, 512, 128],
    ):
        super(DANN, self).__init__()
        
        # ResNet-50 feature extractor (frozen by default)
        self.feature_extractor = resnet
        
        # Class Head
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
        feats = self.feature_extractor(x)


        class_logits = self.class_head(feats)


        if alpha is not None:
            self.grl.set_lambda(alpha)


        reversed_feats = self.grl(feats)
        domain_logits = self.domain_discriminator(reversed_feats)


        return class_logits, domain_logits, feats
    
    def predict(self, x):
        """Predict class labels only (for inference)"""
        with torch.no_grad():
            feats = self.feature_extractor(x)
            logits = self.class_head(feats)
            return torch.argmax(logits, dim=1)
    
    # def unfreeze_backbone(self):
    #     """Unfreeze ResNet-50 backbone for fine-tuning"""
    #     self.feature_extractor.unfreeze()
    
    # def freeze_backbone(self):
    #     """Freeze ResNet-50 backbone"""
    #     self.feature_extractor.freeze()


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
    import math
    p = float(epoch) / float(max(1, max_epochs))
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


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
        gamma: float = 10.0,
        max_grad_norm: Optional[float] = None
    ):
        self.model = model.to(device)
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        
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
            'target_class_loss': [],
            'target_class_acc': []
        }
    
    @staticmethod
    def _unpack_batch(batch):
        """
        Accepts batches that can be (imgs, labels) or (imgs, labels, domains).
        Returns imgs, labels, domains_or_none
        
        """
        # Handle standard tuple format
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                imgs, labels = batch
                domains = None
            elif len(batch) == 3:
                imgs, labels, domains = batch
            else:
                raise ValueError(f"Unsupported batch format with {len(batch)} elements")
        else:
            raise ValueError(f"Batch must be tuple or list, got {type(batch)}")
        
        return imgs, labels, domains
    
    def train_step(self, source_imgs: torch.Tensor, source_labels: torch.Tensor, target_imgs: torch.Tensor, alpha: float) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()

        # move to device
        source_imgs = source_imgs.to(self.device)
        source_labels = source_labels.to(self.device)
        target_imgs = target_imgs.to(self.device)

        # domain labels
        domain_src = torch.zeros(source_imgs.size(0), dtype=torch.long, device=self.device)
        domain_tgt = torch.ones(target_imgs.size(0), dtype=torch.long, device=self.device)

        # forward
        class_out_s, domain_out_s, _ = self.model(source_imgs, alpha)
        _, domain_out_t, _ = self.model(target_imgs, alpha)

        # losses
        class_loss = self.class_criterion(class_out_s, source_labels)
        dom_loss_s = self.domain_criterion(domain_out_s, domain_src)
        dom_loss_t = self.domain_criterion(domain_out_t, domain_tgt)
        domain_loss = dom_loss_s + dom_loss_t
        total_loss = class_loss + 2.0*domain_loss

        total_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # metrics
        with torch.no_grad():
            class_pred = torch.argmax(class_out_s, dim=1)
            class_acc = (class_pred == source_labels).float().mean().item()

            dom_pred_s = torch.argmax(domain_out_s, dim=1)
            dom_pred_t = torch.argmax(domain_out_t, dim=1)
            dom_acc_s = (dom_pred_s == domain_src).float().mean().item()
            dom_acc_t = (dom_pred_t == domain_tgt).float().mean().item()
            dom_acc = 0.5 * (dom_acc_s + dom_acc_t)

        return {
        'class_loss': float(class_loss.item()),
        'domain_loss': float(domain_loss.item()),
        'total_loss': float(total_loss.item()),
        'class_acc': float(class_acc),
        'domain_acc': float(dom_acc)
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
        feats_list = []

        with torch.no_grad():
            pbar = tqdm(target_loader, desc="Evaluating target", leave=False)
            for batch in pbar:
                imgs, labels, _ = self._unpack_batch(batch)
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)

                out = self.model(imgs, 0.0)
                if not isinstance(out, tuple):
                    class_logits = out
                    feats = None
                else:
                    class_logits, _, feats = out

                loss = self.class_criterion(class_logits, labels)
                losses.append(float(loss.item()))

                preds = torch.argmax(class_logits, dim=1)
                y_true.append(labels.cpu())
                y_pred.append(preds.cpu())

                if return_features and feats is not None:
                    feats_list.append(feats.cpu())

        if len(y_true) == 0:
            return {'loss': None, 'accuracy': None, 'y_true': np.array([]), 'y_pred': np.array([]), 'features': None}

        y_true = torch.cat(y_true).numpy()
        y_pred = torch.cat(y_pred).numpy()
        avg_loss = float(np.mean(losses)) if len(losses) > 0 else None
        acc = float((y_true == y_pred).mean()) if len(y_true) > 0 else None

        features_np = None
        if return_features and len(feats_list) > 0:
            features_np = torch.cat(feats_list).numpy()

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
            
            for batch in pbar:
                s_imgs, s_labels, _ = self._unpack_batch(batch)


                try:
                    t_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    t_batch = next(target_iter)


                t_imgs, _, _ = self._unpack_batch(t_batch)

                # match sizes and move to device
                min_b = min(s_imgs.size(0), t_imgs.size(0))
                source_imgs_b = s_imgs[:min_b]
                source_labels_b = s_labels[:min_b]
                target_imgs_b = t_imgs[:min_b]

                # Move
                source_imgs_b = source_imgs_b.to(self.device)
                source_labels_b = source_labels_b.to(self.device)
                target_imgs_b = target_imgs_b.to(self.device)

                metrics = self.train_step(source_imgs_b, source_labels_b, target_imgs_b, alpha)

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

            # Epoch averages
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
        Evaluate DANN model on source vs target loaders.
        Shows classification reports, confusion matrices, and t-SNE feature plots.
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
                for imgs, labels in tqdm(loader, desc=f"Evaluating {name}", leave=False):
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    out = self.model(imgs, 0.0)  # alpha=0 disables gradient reversal

                    if isinstance(out, tuple):
                        cls_logits = out[0]
                        feats = out[2] if len(out) > 2 else None
                    else:
                        cls_logits, feats = out, None

                    preds = torch.argmax(cls_logits, dim=1)
                    y_true.append(labels.cpu())
                    y_pred.append(preds.cpu())

                    if feats is not None:
                        feats_list.append(feats.cpu())

            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
            feats = torch.cat(feats_list).numpy() if feats_list else None

            # --- Classification report
            report = classification_report(
                y_true, y_pred,
                target_names=class_names if class_names else None,
                digits=4
            )
            print(f"\n=== {name.upper()} REPORT ===\n{report}")

            # --- Confusion matrix
            labels_range = np.arange(0, max(y_true.max(), y_pred.max()) + 1)
            cm = confusion_matrix(y_true, y_pred, labels=labels_range)

            if normalize_cm:
                cm_disp = cm.astype(np.float32) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            else:
                cm_disp = cm

            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm_disp,
                annot=False,
                # fmt=".2f" if normalize_cm else "d",
                cmap="Blues",
                xticklabels=False,
                yticklabels=False,
            )
            plt.title(f"Confusion Matrix ({name})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()

            conf_sum = np.sum(cm, axis=1)
            topk_idx = np.argsort(-conf_sum)[:10]
            cm_topk = cm_disp[topk_idx][:, topk_idx]
            labels_topk = [class_names[i] for i in topk_idx]

            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_topk, annot=True, fmt=".2f", cmap="Blues",
                        xticklabels=labels_topk, yticklabels=labels_topk)
            plt.title(f"Top-10 Confusion Matrix ({name})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            plt.show()

            return {"report": report, "cm": cm, "features": feats, "y_true": y_true}

        # --- Evaluate both domains
        src_res = eval_loader("source", src_loader)
        tgt_res = eval_loader("target", tgt_loader)

        # --- t-SNE visualization
        if src_res["features"] is not None and tgt_res["features"] is not None:
            src_feats, tgt_feats = src_res["features"], tgt_res["features"]

            # sample equally
            rng = np.random.RandomState(random_state)
            n_src = min(src_feats.shape[0], tsne_samples // 2)
            n_tgt = min(tgt_feats.shape[0], tsne_samples // 2)

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

        results["source"] = src_res
        results["target"] = tgt_res
        return results
