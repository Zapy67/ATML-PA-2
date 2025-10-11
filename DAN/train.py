import os
import argparse
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange
from sklearn.metrics import classification_report, confusion_matrix
from torchvision.models import resnet152, ResNet152_Weights

# --------------------------
# Utility Functions
# --------------------------

def unfreeze_layers(model, names):
    for name in names:
        module = getattr(model, name)
        for p in module.parameters():
            p.requires_grad = True

def train_step(source_loader, target_loader, model, optimizer, loss_fn, accuracy_fn, device):
    model.to(device)
    model.train()
    
    total_loss = total_supervised = total_mkmmd = correct = 0
    
    for (X_s, Y_s), (X_t, _) in zip(source_loader, target_loader):
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
        correct += accuracy_fn(logits, Y_s)

    n_batches = len(source_loader)
    avg_loss = total_loss / n_batches
    avg_supervised = total_supervised / n_batches
    avg_mkmmd = total_mkmmd / n_batches
    acc = correct / len(source_loader.dataset)
    
    print(f"\nTrain loss: {avg_loss:.5f} | Supervised: {avg_supervised:.5f} | MK-MMD: {avg_mkmmd:.5f} | Train acc: {acc*100:.2f}%\n", flush=True)
    return avg_loss, avg_supervised, avg_mkmmd, acc

def loader_accuracy(loader, model, accuracy_fn, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits, _ = model(X)
            correct += accuracy_fn(logits, Y)
    return correct / len(loader.dataset)

def train(source_loader, target_loader, test_loader, epochs, optimizer, model, loss_fn, accuracy_fn, device):
    source_losses = []
    supervised_losses = []
    mkmmd_losses = []
    train_accs = []
    test_accs = []

    for epoch in trange(epochs, desc="Training"):
        print(f"\nEpoch {epoch+1}/{epochs}")
        avg_loss, avg_supervised, avg_mkmmd, avg_acc = train_step(
            source_loader, target_loader, model, optimizer, loss_fn, accuracy_fn, device
        )

        source_losses.append(avg_loss)
        supervised_losses.append(avg_supervised)
        mkmmd_losses.append(avg_mkmmd)
        train_accs.append(avg_acc)

        test_acc = loader_accuracy(test_loader, model, accuracy_fn, device)
        test_accs.append(test_acc)

        print(f"Epoch {epoch+1}: Train loss={avg_loss:.5f}, Train acc={avg_acc*100:.2f}%, Test acc={test_acc*100:.2f}%")

    return source_losses, supervised_losses, mkmmd_losses, train_accs, test_accs

def evaluate_loader(loader, model, device, class_names=None, savepath=None):
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits, _ = model(X)
            all_preds.append(torch.argmax(logits, dim=1).cpu())
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
   
    if savepath:
        plt.savefig(savepath)
    plt.show()

    return report, cm

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DAN Training Script")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--save_path", type=str, default="dan_model.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------
    # Model
    # ----------------------
    from architecture import dan_resnet, DANLoss, _forward_impl  # or define _forward_impl here
    import dataset  # assumes you have dataset.source_loader/target_loader defined

    dan_resnet = resnet152(ResNet152_Weights.DEFAULT)
    dan_resnet.forward = _forward_impl

    for p in dan_resnet.parameters():
        p.requires_grad = False
    unfreeze_layers(dan_resnet, ['layer3','layer4', 'fc' ])

    # ----------------------
    # Optimizer & Loss
    # ----------------------
    loss_fn = DANLoss(sigmas=[1.0, 5.0, 10.0], scale=1.0)
    optimizer = torch.optim.SGD(dan_resnet.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # ----------------------
    # Training
    # ----------------------
    source_loader = dataset.source_loader
    target_loader = dataset.target_loader

    # define accuracy function
    def accuracy_fn(logits, labels):
        return (torch.argmax(logits, dim=1) == labels).sum().item()

    source_losses, supervised_losses, mkmmd_losses, train_accs, test_accs = train(
        source_loader, target_loader, target_loader,
        epochs=args.epochs,
        optimizer=optimizer,
        model=dan_resnet,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

    # ----------------------
    # Plot
    # ----------------------
    fig, axes= plt.subplots(1,2)
    fig.suptitle("Training Dynamics")
    fig.set_size_inches(10, 3)
    ax1, ax2 = axes

    ax1.set(title="Accuracies", xlabel="Epoch", ylabel="Accuracy")
    ax1.plot(train_accs, label="Source")
    ax1.plot(test_accs, label="Target")
    ax1.legend(), ax1.grid(True)

    ax2.set(title="Source Losses", xlabel="Epoch", ylabel="Loss")
    ax2.plot(source_losses, label="Total")
    ax2.plot(supervised_losses, label="Supervised")
    ax2.plot(mkmmd_losses, label="MKMMD")
    ax2.legend(), ax2.grid(True)

    os.makedirs('figs', exist_ok=True)
    plt.savefig('figs/training_dynamics.png')
    torch.save(dan_resnet.state_dict(), args.save_path)