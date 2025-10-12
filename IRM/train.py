import torch
from architecture import IRMLoss
import tqdm

def accuracy_fn(data: torch.Tensor, labels: torch.Tensor):
    return torch.sum(data.argmax(dim=1) == labels)


def loader_accuracy(loader, model, accuracy_fn, device):
    model.eval()
    correct = 0
    with torch.inference_mode():
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            correct += accuracy_fn(model(X), Y)
    return correct / len(loader.dataset)


def train_step(train_loaders, model, optimizer, loss_fn, accuracy_fn, device):
    model.to(device)
    model.train()

    dummy_w = torch.nn.Parameter(torch.tensor([1.0])).to(device)
    total_erm_loss = 0.0
    total_penalty = 0.0
    correct = 0
    total = 0
    batch_idx = 1

    for batches in zip(*train_loaders):
        optimizer.zero_grad()
        error = 0.0
        penalty = 0.0

        for data, target in batches:
            data, target = data.to(device).float(), target.to(device).long()
            output = model(data)
            erm, penalty = loss_fn(output, target, dummy_w)
            error += erm
            correct += accuracy_fn(output, target)
            total += len(output)

        (error + penalty).backward()
        optimizer.step()

        total_erm_loss += error.item()
        total_penalty += penalty.item()
        batch_idx += 1

    return total_erm_loss / batch_idx, total_penalty / batch_idx, correct / total


def train(train_loaders, val_loader, epochs, optimizer, model, loss_fn, accuracy_fn, device):
    model.to(device)

    train_accs = []
    erm_losses, penalties = [], []
    test_accs = []

    for epoch in tqdm.trange(epochs, desc="Training"):
        print(f"\nEpoch {epoch+1}/{epochs}")

        avg_erm_loss, avg_penalty, avg_acc = train_step(
            train_loaders, model, optimizer, loss_fn, accuracy_fn, device
        )

        train_accs.append(avg_acc)
        erm_losses.append(avg_erm_loss)
        penalties.append(avg_penalty)

        test_acc = loader_accuracy(val_loader, model, accuracy_fn, device)
        test_accs.append(test_acc)

        print(
            f'\tERM loss: {avg_erm_loss:.6f}\t'
            f'Grad penalty: {avg_penalty:.6f}\t'
            f'Train Accuracy: {avg_acc:.4f}\t'
            f'Test Accuracy: {test_acc:.4f}'
        )

    return train_accs, test_accs, erm_losses, penalties


def train_workflow(model, src_datasets, tgt_dataset, config, device):
    epochs         = config.get('epochs', 5)
    lr             = config.get('lr', 1e-5)
    weight_decay   = config.get('weight_decay', 0.0)
    batch_size     = config.get('batch_size', 32)
    momentum       = config.get('momentum', 0.0)
    phi          = config.get('phi', 0.5)

    loss_fn = IRMLoss(phi)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        momentum=momentum
    )
    source_loaders = []
    for ds in src_datasets:
        source_loaders.append(torch.utils.data.DataLoader(ds, batch_size, shuffle=True))


    target_loader = torch.utils.data.DataLoader(tgt_dataset, batch_size, shuffle=True)

    train(
        source_loaders,
        target_loader,
        epochs=epochs,
        optimizer=optimizer,
        model=model,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )
    
  

