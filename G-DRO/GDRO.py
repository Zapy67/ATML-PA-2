"""
GDRO Trainer for Office-Home using dataset utilities from dataset.py
- GDROTrainer class: load data (via dataset.make_domain_subdatasets), build model, train with G-DRO, early stopping, save/load
- analysis(...) method: classification report (sklearn) and t-SNE visualization of embeddings

Note: all dataset logic lives in dataset.py. This file imports helpers from that module.
"""

import os
import math
import random
from collections import defaultdict

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models

from sklearn.metrics import classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# import dataset helpers (everything dataset-related lives in dataset.py)
from utils.dataset import (
    make_domain_subdatasets,
    prepare_dataframe,
    truncate_up_to,
)

class GDROTrainer:
    def __init__(
        self,
        csv_file,
        root_dir,
        domains=['Art', 'Clipart', 'Product', 'Real World'],
        img_size=224,
        batch_size=32,
        num_workers=2,
        lr=3e-4,
        weight_decay=1e-3,
        eta_q=0.5,
        ema_alpha=0.2,
        num_epochs=20,
        patience=5,
        seed=42,
        device=None,
        test_frac=0.1,
    ):
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.domains = list(domains)
        self.img_size = img_size
        self.batch_size = batch_size
        self.sub_bs = max(1, batch_size // max(1, len(self.domains)))
        self.num_workers = num_workers
        self.lr = lr
        self.weight_decay = weight_decay
        self.eta_q = eta_q
        self.ema_alpha = ema_alpha
        self.num_epochs = num_epochs
        self.patience = patience
        self.seed = seed
        self.test_frac = test_frac

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # placeholders
        self.class_to_idx = None
        self.classes = None
        self.domain_train_datasets = None
        self.domain_test_datasets = None
        self.train_loaders = None
        self.test_items = None
        self.steps_per_epoch = None

        self.model = None
        self.optimizer = None
        self.criterion = None

        # G-DRO state
        self.num_groups = len(self.domains)
        self.q = None
        self.running_group_loss = None
        self.group_seen_counts = None

        self.best_worst = -1.0
        self.patience_counter = 0

        # transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ---------------------------
    # Data preparation: delegates to dataset.make_domain_subdatasets
    # ---------------------------
    def build_datasets_and_loaders(self):
        (self.domain_train_datasets,
         self.domain_test_datasets,
         self.class_to_idx,
         self.steps_per_epoch,
         self.test_items,
         self.classes) = make_domain_subdatasets(
            csv_file=self.csv_file,
            root_dir=self.root_dir,
            domains=self.domains,
            train_transform=self.train_transform,
            test_transform=self.test_transform,
            test_frac=self.test_frac,
            seed=self.seed,
            img_size=self.img_size,
            sub_bs=self.sub_bs,
            num_workers=self.num_workers,
        )

        # loaders per domain with balanced sub-batch
        self.train_loaders = [
            DataLoader(td, batch_size=self.sub_bs, shuffle=True, drop_last=True,
                       num_workers=self.num_workers, pin_memory=True)
            for td in self.domain_train_datasets
        ]
        return self.train_loaders, self.test_items

    # ---------------------------
    # Model
    # ---------------------------
    def build_model(self, base_model='resnet50', pretrained=True):
        if base_model == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            model.fc = nn.Linear(model.fc.in_features, len(self.class_to_idx) if self.class_to_idx is not None else 1000)
        else:
            raise NotImplementedError('only resnet50 is implemented in this helper')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # initialize gdro state
        self.q = torch.ones(self.num_groups, device=self.device) / float(self.num_groups)
        self.running_group_loss = torch.zeros(self.num_groups, device=self.device)
        self.group_seen_counts = torch.zeros(self.num_groups, device=self.device)
        return self.model

    # ---------------------------
    # Training loop (G-DRO)
    # ---------------------------
    def train(self, num_epochs=None, model_out='g_dro_officehome.pt'):
        if num_epochs is None:
            num_epochs = self.num_epochs
        if self.train_loaders is None or self.test_items is None:
            print("Building Dataset and Loaders")
            self.build_datasets_and_loaders()
        if self.model is None:
            print("Building Model")
            self.build_model()

        loaders = self.train_loaders
        iters = [iter(dl) for dl in loaders]
        steps_per_epoch = self.steps_per_epoch

        best_worst = self.best_worst
        patience_counter = self.patience_counter

        for epoch in range(1, num_epochs + 1):
            self.model.train()
            epoch_losses = []
            with tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{num_epochs}") as pbar:
                for step in pbar:
                    # fetch one batch from each domain
                    batches = []
                    for i, it in enumerate(iters):
                        try:
                            batch = next(it)
                        except StopIteration:
                            iters[i] = iter(loaders[i])
                            batch = next(iters[i])
                        batches.append(batch)

                    imgs_list = []
                    labels_list = []
                    for imgs_b, labels_b in batches:
                        imgs_list.append(imgs_b)
                        labels_list.append(labels_b)

                    imgs = torch.cat(imgs_list, dim=0).to(self.device)
                    labels = torch.cat(labels_list, dim=0).to(self.device)

                    groups_list = [torch.full((imgs_b.size(0),), d, dtype=torch.long) for d, imgs_b in enumerate(imgs_list)]
                    groups = torch.cat(groups_list, dim=0).to(self.device)

                    logits = self.model(imgs)
                    per_sample = self.criterion(logits, labels)

                    # compute mean loss per domain present
                    batch_group_losses = torch.zeros(self.num_groups, device=self.device)
                    present = torch.zeros(self.num_groups, dtype=torch.bool, device=self.device)
                    for g_id in range(self.num_groups):
                        mask = (groups == g_id)
                        if mask.sum() == 0:
                            continue
                        lg = per_sample[mask].mean().detach()
                        batch_group_losses[g_id] = lg
                        present[g_id] = True
                        # update running EMA
                        self.running_group_loss[g_id] = (1.0 - self.ema_alpha) * self.running_group_loss[g_id] + self.ema_alpha * lg
                        self.group_seen_counts[g_id] += 1

                    used_group_losses = batch_group_losses.clone()
                    for gi in range(self.num_groups):
                        if not present[gi]:
                            if self.group_seen_counts[gi] > 0:
                                used_group_losses[gi] = self.running_group_loss[gi].detach()
                            else:
                                used_group_losses[gi] = torch.tensor(0.0, device=self.device)

                    # multiplicative weights update
                    q_np = (self.q * torch.exp(self.eta_q * used_group_losses.detach()))
                    self.q = q_np / (q_np.sum() + 1e-12)

                    # per-sample weights and final loss
                    w_i = self.q[groups]
                    w_i = w_i / (w_i.sum() + 1e-12)
                    loss = (w_i * per_sample).sum()

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    epoch_losses.append(loss.item())
                    pbar.set_postfix({'loss': f"{np.mean(epoch_losses):.4f}", 'worst_q': int(torch.argmax(self.q).item())})

            # testing at epoch end
            test_stats = self.evaluate()
            worst_acc = test_stats['worst_group_acc']
            print(f"Epoch {epoch} testing: worst_group_acc={worst_acc:.4f}, per_group={test_stats['per_group_acc']}")

            if worst_acc is not None and worst_acc > best_worst:
                best_worst = worst_acc
                self.save(model_out)
                print(f"  Saved best model (worst={best_worst:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"Early stopping: no improvement in worst-group test for {self.patience} epochs.")
                    break

        # update internal state
        self.best_worst = best_worst
        self.patience_counter = patience_counter
        print("Training finished. Best worst-group acc:", best_worst)

    # ---------------------------
    # Evaluation helpers
    # ---------------------------
    def evaluate(self):
        self.model.eval()
        group_correct = defaultdict(int)
        group_total = defaultdict(int)
        losses = []
        criterion = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            for it in range(0, len(self.test_items), self.batch_size):
                batch = self.test_items[it:it + self.batch_size]
                imgs_b = torch.stack([b[0] for b in batch]).to(self.device)
                lbls_b = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
                groups_b = torch.tensor([b[2] for b in batch], dtype=torch.long).to(self.device)
                out = self.model(imgs_b)
                preds = out.argmax(dim=1)
                per = criterion(out, lbls_b).cpu()
                losses.append(per)
                for i in range(len(preds)):
                    gi = int(groups_b[i].item())
                    group_total[gi] += 1
                    if int(preds[i].cpu().item()) == int(lbls_b[i].cpu().item()):
                        group_correct[gi] += 1
        per_group_acc = {}
        for g in range(self.num_groups):
            per_group_acc[g] = (group_correct[g] / group_total[g]) if group_total[g] > 0 else None
        all_losses = torch.cat(losses).numpy() if len(losses) > 0 else np.array([])
        avg_loss = float(all_losses.mean()) if all_losses.size > 0 else None
        worst_group_acc = min([a for a in per_group_acc.values() if a is not None]) if any(v is not None for v in per_group_acc.values()) else None
        return {'per_group_acc': per_group_acc, 'avg_loss': avg_loss, 'worst_group_acc': worst_group_acc}

    def predict_on_test(self):
        self.model.eval()
        preds_all = []
        labels_all = []
        groups_all = []
        with torch.no_grad():
            for it in range(0, len(self.test_items), self.batch_size):
                batch = self.test_items[it:it + self.batch_size]
                imgs_b = torch.stack([b[0] for b in batch]).to(self.device)
                lbls_b = torch.tensor([b[1] for b in batch], dtype=torch.long).to(self.device)
                groups_b = torch.tensor([b[2] for b in batch], dtype=torch.long).to(self.device)
                out = self.model(imgs_b)
                preds = out.argmax(dim=1)
                preds_all.extend([int(p.cpu().item()) for p in preds])
                labels_all.extend([int(l.cpu().item()) for l in lbls_b])
                groups_all.extend([int(g.cpu().item()) for g in groups_b])
        return preds_all, labels_all, groups_all

    # ---------------------------
    # Analysis: classification report and t-SNE
    # ---------------------------
    def analysis(self, out_dir='analysis_out', tsne_samples=2000, perplexity=30, n_iter=1000):
        os.makedirs(out_dir, exist_ok=True)
        preds, labels, groups = self.predict_on_test()

        # classification report (global)
        target_names = [c for c in sorted(self.class_to_idx, key=lambda k: self.class_to_idx[k])]
        report = classification_report(labels, preds, target_names=target_names, zero_division=0)
        report_path = os.path.join(out_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        print("Saved classification report to:", report_path)

        # compute embeddings (penultimate layer) for test set
        embeddings = []
        emb_labels = []
        emb_groups = []
        self.model.eval()
        trunk = nn.Sequential(*list(self.model.children())[:-1]).to(self.device).eval()
        with torch.no_grad():
            for it in range(0, len(self.test_items), self.batch_size):
                batch = self.test_items[it:it + self.batch_size]
                imgs_b = torch.stack([b[0] for b in batch]).to(self.device)
                lbls_b = [b[1] for b in batch]
                groups_b = [b[2] for b in batch]
                feats = trunk(imgs_b)
                feats = torch.flatten(feats, 1).cpu().numpy()
                embeddings.append(feats)
                emb_labels.extend(lbls_b)
                emb_groups.extend(groups_b)
        embeddings = np.vstack(embeddings)
        emb_labels = np.array(emb_labels)
        emb_groups = np.array(emb_groups)

        # subsample for t-SNE if too large
        n = embeddings.shape[0]
        if n == 0:
            print("No test embeddings available for t-SNE.")
            return
        sample_idx = np.arange(n)
        if n > tsne_samples:
            rng = np.random.RandomState(self.seed)
            sample_idx = rng.choice(n, size=tsne_samples, replace=False)
        emb_subset = embeddings[sample_idx]
        labels_subset = emb_labels[sample_idx]
        groups_subset = emb_groups[sample_idx]

        print(f"Running t-SNE on {emb_subset.shape[0]} samples (this may take a while)...")
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=self.seed)
        z = tsne.fit_transform(emb_subset)

        # scatter colored by domain (group)
        plt.figure(figsize=(8, 6))
        num_groups = len(self.domains)
        for g in range(num_groups):
            mask = (groups_subset == g)
            if mask.sum() == 0:
                continue
            plt.scatter(z[mask, 0], z[mask, 1], label=self.domains[g], alpha=0.6, s=6)
        plt.legend()
        plt.title('t-SNE of test embeddings (colored by domain)')
        tsne_path = os.path.join(out_dir, 'tsne_domains.png')
        plt.show()
        # plt.savefig(tsne_path, dpi=150, bbox_inches='tight')
        # plt.close()
        print("Saved t-SNE (domains) to:", tsne_path)

        # scatter colored by predicted label (top-10 frequent labels only to keep plot readable)
        unique_labels, counts = np.unique(labels_subset, return_counts=True)
        topk = 10
        topk_idx = np.argsort(counts)[-topk:][::-1] if len(counts) > topk else np.arange(len(counts))
        top_labels = unique_labels[topk_idx]
        plt.figure(figsize=(8, 6))
        for lab in top_labels:
            mask = (labels_subset == lab)
            plt.scatter(z[mask, 0], z[mask, 1], label=self.classes[lab] if lab < len(self.classes) else str(lab), alpha=0.6, s=6)
        plt.legend(fontsize='small')
        plt.title('t-SNE of test embeddings (top labels)')
        tsne_labels_path = os.path.join(out_dir, 'tsne_top_labels.png')
        plt.show()
        # plt.savefig(tsne_labels_path, dpi=150, bbox_inches='tight')
        # plt.close()
        print("Saved t-SNE (labels) to:", tsne_labels_path)

    # ---------------------------
    # Save / load
    # ---------------------------
    def save(self, path):
        state = {'model_state_dict': self.model.state_dict(),
                 'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer is not None else None,
                 'q': self.q.cpu().numpy() if self.q is not None else None,
                 'class_to_idx': self.class_to_idx,
                 'domains': self.domains}
        torch.save(state, path)

    def load(self, path, map_location=None):
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        if self.model is None:
            self.build_model()
        self.model.load_state_dict(state['model_state_dict'])
        if state.get('optimizer_state_dict') is not None and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
            except Exception:
                pass
        if state.get('q') is not None:
            self.q = torch.tensor(state['q'], device=self.device)
        if state.get('class_to_idx') is not None:
            self.class_to_idx = state['class_to_idx']


# End of file
