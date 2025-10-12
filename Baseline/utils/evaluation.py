import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import seaborn as sns

def evaluate_ds(ds, model, device, domain_name=None, class_names=None):
    model.eval()
    all_preds = []
    all_labels = []

    loader = torch.utils.data.DataLoader(ds, shuffle=False, batch_size=32)

    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)  
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(Y.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    if domain_name:
        print(f"\n=== Evaluation for {domain_name} Domain ===")
    print(f"Accuracy: {acc*100:.2f}%")
    print("Confusion Matrix:\n")
    sns.heatmap(cm, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    print("Classification Report:\n", report)

    return {
        'domain': domain_name,
        'accuracy': acc,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred
    }