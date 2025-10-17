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
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
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
        pretrained: bool = True,
        freeze_backbone: bool = False,
        class_head_dims: list = None,
        domain_discriminator_dims: list = [1024, 512],
    ):
        super(DANN, self).__init__()
        
        # ResNet-50 feature extractor (frozen by default)
        self.feature_extractor = ResNet50FeatureExtractor(
            pretrained=pretrained,
            freeze=freeze_backbone
        )
        
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
        total_loss = class_loss + domain_loss

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
        domain_loaders: Dict[str, DataLoader],
        source_domains: list,
        target_domains: list,
        class_names: Optional[list] = None,
        tsne_perplexity: int = 30,
        tsne_samples: Optional[int] = 2000,
        pca_before_tsne: bool = True,
        random_state: int = 0,
        normalize_cm: bool = True,
    ):
        """
        Analysis across multiple domain dataloaders.

        Args:
            domain_loaders: dict mapping domain_name -> DataLoader
            source_domains: list of domain_name(s) considered "source" (for t-SNE grouping)
            target_domains: list of domain_name(s) considered "target" (for t-SNE grouping)
            class_names: optional list of class names (for readable reports and CM ticks)
            tsne_perplexity, tsne_samples, pca_before_tsne, random_state: TSNE options
            normalize_cm: if True, show confusion matrix as row-normalized (per-true-class rates)

        Returns:
            results: dict with keys:
                'per_domain_report' : {domain_name: classification_report(str)}
                'per_domain_cm'     : {domain_name: confusion_matrix (np.array)}
                'tsne'              : { 'embeddings': np.array (N,2), 'domains': list(domain_names), 'labels': np.array(classes) }
        """
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        import torch

        self.model.eval()

        per_domain_report = {}
        per_domain_cm = {}
        features_by_domain = {}
        ytrue_by_domain = {}
        ypred_by_domain = {}

        # 1) Collect predictions, labels, and features per-domain
        for domain_name, loader in domain_loaders.items():
            y_true = []
            y_pred = []
            feats_list = []

            with torch.no_grad():
                pbar = tqdm(loader, desc=f"Collecting [{domain_name}]", leave=False)
                for batch in pbar:
                    imgs, labels, *rest = batch if len(batch) >= 2 else (batch[0], batch[1])
                    imgs = imgs.to(self.device)
                    labels = labels.to(self.device)

                    out = self.model(imgs, 0.0)
                    if isinstance(out, tuple):
                        class_logits = out[0]
                        feats = out[2] if len(out) > 2 else None  # 3rd element
                    else:
                        class_logits = out
                        feats = None

                    preds = torch.argmax(class_logits, dim=1)

                    y_true.append(labels.cpu())
                    y_pred.append(preds.cpu())
                    if feats is not None:
                        feats_list.append(feats.detach().cpu())

            if len(y_true) == 0:
                # empty loader
                per_domain_report[domain_name] = "EMPTY_LOADER"
                per_domain_cm[domain_name] = None
                features_by_domain[domain_name] = None
                ytrue_by_domain[domain_name] = np.array([])
                ypred_by_domain[domain_name] = np.array([])
                continue

            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
            ytrue_by_domain[domain_name] = y_true
            ypred_by_domain[domain_name] = y_pred

            # features
            if len(feats_list) > 0:
                feats_np = torch.cat(feats_list).numpy()
                features_by_domain[domain_name] = feats_np
            else:
                features_by_domain[domain_name] = None

            # classification report
            if class_names is not None:
                report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
            else:
                report = classification_report(y_true, y_pred, digits=4)
            per_domain_report[domain_name] = report

            # confusion matrix
            # choose label order: if class_names provided, use range(len(class_names))
            if class_names is not None:
                labels = list(range(len(class_names)))
            else:
                # attempt to include all classes present in y_true or y_pred
                labels = np.arange(0, max(y_true.max() if y_true.size else 0, y_pred.max() if y_pred.size else 0) + 1)
                if labels.size == 0:
                    labels = np.array([0])

            cm = confusion_matrix(y_true, y_pred, labels=labels)
            if normalize_cm:
                # normalize rows (true class -> predicted distribution)
                row_sums = cm.sum(axis=1, keepdims=True).astype(np.float32)
                row_sums[row_sums == 0] = 1.0
                cm_display = cm.astype(np.float32) / row_sums
            else:
                cm_display = cm

            per_domain_cm[domain_name] = cm  # store raw cm

            # plot heatmap
            plt.figure(figsize=(6,5))
            sns.heatmap(cm_display, annot=True, fmt='.2f' if normalize_cm else 'd',
                        xticklabels=(class_names if class_names is not None else labels),
                        yticklabels=(class_names if class_names is not None else labels),
                        cmap='Blues')
            plt.title(f'Confusion Matrix ({domain_name})')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.show()

            # print report
            print(f"\n=== Classification report for domain: {domain_name} ===")
            print(report)

        # 2) t-SNE comparing source vs target domains
        # Validate provided domain names
        all_domain_names = list(domain_loaders.keys())
        for name in list(source_domains) + list(target_domains):
            if name not in domain_loaders:
                raise ValueError(f"Domain '{name}' not present in domain_loaders keys: {all_domain_names}")

        # Build features and labels for t-SNE. We'll sample to tsne_samples per entire plot.
        tsne_feats = []
        tsne_domain_labels = []
        tsne_class_labels = []

        # helper: sample from each domain up to per_domain_quota
        total_domains = len(source_domains) + len(target_domains)
        if tsne_samples is None or tsne_samples <= 0:
            per_domain_quota = None
        else:
            per_domain_quota = max(1, tsne_samples // max(1, total_domains))

        rng = np.random.RandomState(random_state)

        def gather_for_names(names, tag):
            # tag is string like 'source' or 'target' to help in plotting legend
            for dname in names:
                feats = features_by_domain.get(dname, None)
                y_true = ytrue_by_domain.get(dname, None)
                if feats is None or feats.size == 0:
                    continue
                n = feats.shape[0]
                if per_domain_quota is None:
                    idxs = np.arange(n)
                else:
                    if n <= per_domain_quota:
                        idxs = np.arange(n)
                    else:
                        idxs = rng.choice(n, per_domain_quota, replace=False)
                tsne_feats.append(feats[idxs])
                tsne_domain_labels += [(dname, tag)] * len(idxs)
                if y_true is not None and y_true.size > 0:
                    tsne_class_labels.append(y_true[idxs])
                else:
                    tsne_class_labels.append(np.full(len(idxs), -1, dtype=int))

        gather_for_names(source_domains, 'source')
        gather_for_names(target_domains, 'target')

        if len(tsne_feats) == 0:
            print("No features available for t-SNE (ensure model returns features during inference).")
            tsne_result = {'embeddings': None, 'domains': None, 'labels': None}
        else:
            X = np.vstack(tsne_feats)
            domain_meta = tsne_domain_labels  # list of (domain_name, 'source'/'target') repeated per sample
            class_meta = np.concatenate(tsne_class_labels)

            # optional PCA
            if pca_before_tsne and X.shape[1] > 50:
                pca = PCA(n_components=50, random_state=random_state)
                X_reduced = pca.fit_transform(X)
            else:
                X_reduced = X

            tsne = TSNE(n_components=2, perplexity=tsne_perplexity, init='pca', random_state=random_state)
            emb = tsne.fit_transform(X_reduced)

            # build arrays for plotting
            domain_names = [d for (d, tag) in domain_meta]
            domain_tags = [tag for (d, tag) in domain_meta]
            unique_domains = list(dict.fromkeys(domain_names))  # preserve order
            unique_tags = ['source', 'target']

            # color by domain name
            palette = sns.color_palette('tab10', n_colors=max(10, len(unique_domains)))
            domain_to_color = {d: palette[i % len(palette)] for i, d in enumerate(unique_domains)}
            tag_to_marker = {'source': 'o', 'target': 'X'}

            plt.figure(figsize=(10, 8))
            for i in range(emb.shape[0]):
                dn = domain_names[i]
                tag = domain_tags[i]
                plt.scatter(emb[i,0], emb[i,1], marker=tag_to_marker[tag], color=domain_to_color[dn], s=12, alpha=0.8)

            # build legend handles
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
            domain_handles = [mpatches.Patch(color=domain_to_color[d], label=d) for d in unique_domains]
            tag_handles = [Line2D([0],[0], marker=tag_to_marker[t], color='w', markerfacecolor='k', markersize=8, label=t) for t in unique_tags]
            plt.legend(handles=domain_handles + tag_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title(f"t-SNE of features: sources={source_domains}  targets={target_domains}")
            plt.tight_layout()
            plt.show()

            tsne_result = {
                'embeddings': emb,
                'domain_names': domain_names,
                'domain_tags': domain_tags,
                'class_labels': class_meta
            }

        results = {
            'per_domain_report': per_domain_report,
            'per_domain_cm': per_domain_cm,
            'tsne': tsne_result
        }

        return results