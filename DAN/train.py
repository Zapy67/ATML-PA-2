import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import resnet152, ResNet152_Weights
from architecture import DANLoss

# --------------------------
# Utility Functions
# --------------------------

import torch
from tqdm import trange

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import seaborn as sns


def accuracy_fn(logits, labels):
    return (torch.argmax(logits, dim=1) == labels).sum().item()

def train_step(source_loader, target_loader, model, optimizer, loss_fn, accuracy_fn, device):
    model.train()
    
    total_supervised = 0
    total_mkmmd = 0
   
    for (X_s, Y_s), (X_t, _) in zip(source_loader, target_loader):
        X_s, Y_s, X_t = X_s.to(device), Y_s.to(device), X_t.to(device)
        
        optimizer.zero_grad()
        logits, source_features = model(X_s)
        _, target_features = model(X_t)
        
        supervised, scaled_mkmmd = loss_fn(source_features, target_features, logits, Y_s)
        loss = supervised + scaled_mkmmd
        loss.backward()
        optimizer.step()
       
        total_supervised += supervised.item()
        total_mkmmd += scaled_mkmmd.item()

    n = len(source_loader.dataset)
    
    avg_supervised = total_supervised / n
    avg_mkmmd = total_mkmmd / n
    avg_loss = avg_supervised + avg_mkmmd
    
    return avg_loss, avg_supervised, avg_mkmmd


def train(source_loader, target_loader, test_loader, epochs, optimizer, model, loss_fn, accuracy_fn, device):
    source_losses = []
    supervised_losses = []
    mkmmd_losses = []
    train_accs = []
    test_accs = []

    for epoch in trange(epochs, desc="Training"):
        print(f"\nEpoch {epoch+1}/{epochs}")  
        loss_fn.scale = min(loss_fn.scale * 1.25, 512)
        model.to(device)

        avg_loss, avg_supervised, avg_mkmmd, = train_step(
            source_loader, target_loader, model, optimizer, loss_fn, accuracy_fn, device
        )
        test_acc_target = evaluate_accuracy(test_loader, model, accuracy_fn, device)
        train_acc_source = evaluate_accuracy(source_loader, model, accuracy_fn, device)

        source_losses.append(avg_loss)
        supervised_losses.append(avg_supervised)
        mkmmd_losses.append(avg_mkmmd)
        train_accs.append(train_acc_source)
        
        test_accs.append(test_acc_target)
        print(f"\nTrain loss: {avg_loss:.5f} | Supervised: {avg_supervised:.5f} | MK-MMD: {avg_mkmmd:.5f} | Source train acc: {train_acc_source*100:.2f}% | Target test acc={test_acc*100:.2f}%\n", flush=True)

    return source_losses, supervised_losses, mkmmd_losses, train_accs, test_accs

def evaluate_accuracy(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)  
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(Y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    return accuracy_score(y_true, y_pred)

def train_workflow(model, src_dataset, tgt_dataset, val_dataset, config, device):
    epochs         = config.get('epochs', 5)
    lr             = config.get('lr', 1e-5)
    weight_decay   = config.get('weight_decay', 0.0)
    batch_size     = config.get('batch_size', 32)
    momentum       = config.get('momentum', 0.0)
    sigmas         = config.get('sigmas', [1.0])
    scale          = config.get('scale', [1.0])
    
    loss_fn = DANLoss(sigmas=sigmas, scale=scale)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    source_loader = torch.utils.data.DataLoader(src_dataset, batch_size, shuffle=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size, shuffle=True, drop_last=True)
    test_loader   = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    
    train(source_loader, target_loader, test_loader, epochs, optimizer, model, loss_fn, accuracy_fn, device)

    
    


