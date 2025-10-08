import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import tqdm

def unfreeze_layers(model, names):
     for name in names:
          module = getattr(model, name)
          for p in module.parameters():
               p.requires_grad=True

def train_step(source_loader, target_loader, model, optimizer, loss_fn, accuracy_fn, device):
    model.to(device)
    model.train()
    
    total_loss = 0
    total_supervised = 0
    total_mkmmd = 0
    correct = 0
    
    for (X_s, Y_s), (X_t,Y_t) in zip(source_loader, target_loader):  # ignore target labels
        X_s, Y_s, X_t = X_s.to(device), Y_s.to(device), X_t.to(device)
        optimizer.zero_grad()

        logits, source_features = model(X_s)
        _, target_features = model(X_t)

        supervised, scaled_mkmmd = loss_fn(source_features, target_features, logits, Y_s)
        loss = supervised + scaled_mkmmd
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_supervised += supervised.item()
        total_mkmmd += scaled_mkmmd.item()
        correct += accuracy_fn(source_features, Y_s)

    n_batches = len(source_loader)
    avg_loss = total_loss / n_batches
    avg_supervised = total_supervised / n_batches
    avg_mkmmd = total_mkmmd / n_batches
    acc = correct / len(source_loader.dataset)

    print(f"\nTrain loss: {avg_loss:.5f} | Supervised: {avg_supervised:.5f} | MK-MMD: {avg_mkmmd:.5f} | Train acc: {acc*100:.2f}%\n", flush=True)
    return avg_loss,avg_supervised, avg_mkmmd, acc

def loader_accuracy(loader, model, accuracy_fn, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            correct += accuracy_fn(model(X), Y)
            
    return correct / len(loader.dataset)
    

def train(source_loader, target_loader, test_loader, epochs, optimizer, model, loss_fn, accuracy_fn, device):
    model.to(device)
    
    source_losses, source_accs = [], []
    supervised_losses, mkmmd_losses = [], []
    test_accs = []

    for epoch in tqdm.trange(epochs, desc="Training"):
        print(f"\nEpoch {epoch+1}/{epochs}")

        avg_loss, avg_supervised, avg_mkmmd, avg_acc = train_step(source_loader, target_loader, model, optimizer, loss_fn, accuracy_fn, device)

        source_losses.append(avg_loss)
        source_accs.append(avg_acc)
        supervised_losses.append(avg_supervised)
        mkmmd_losses.append(avg_mkmmd)

        test_acc = loader_accuracy(test_loader, model, accuracy_fn, device)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}: Train loss={avg_loss:.5f}, Train acc={avg_acc*100:.2f}%, Test acc={test_acc*100:.2f}%")

    return source_losses, supervised_losses, mkmmd_losses, source_accs, test_accs


def evaluate_loader(loader, model, device, class_names=None, savepath=None):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)

            all_preds.append(torch.argmax(preds, dim=1).cpu())
            all_labels.append(Y.cpu())

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix')
   
    if savepath is not None:
        plt.savefig(savepath)
    plt.show()

    return report, cm



