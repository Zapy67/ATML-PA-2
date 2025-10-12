import torch
import tqdm


def accuracy_fn(preds, labels):
    return torch.sum(preds.argmax(dim=1) == labels).item()

def train_step(train_loader, model, optimizer, loss_fn, accuracy_fn, device):
    model.train()
    train_loss = 0 
    num_correct = 0

    for X,Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        preds = model(X)
        loss = loss_fn(preds,Y)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        num_correct += accuracy_fn(preds,Y)

    n = len(train_loader.dataset)
    avg_loss = train_loss/n 
    acc = num_correct/n
    print(f"\nTrain loss: {avg_loss:.5f} |  Train acc: {acc*100:.2f} %\n", flush=True)
    return avg_loss, acc

def test_step(test_loader, model, loss_fn, accuracy_fn, device):
    model.eval()
    test_loss = 0
    num_correct = 0

    with torch.inference_mode():
        for X,Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            preds = model(X)
            test_loss += loss_fn(preds,Y).item()
            num_correct += accuracy_fn(preds, Y)

    n = len(test_loader.dataset)
    avg_loss = test_loss/n #loss per sample
    acc = num_correct/n
    print(f"Test loss: {avg_loss:.5f} | Test acc: {acc*100:.2f} %\n", flush=True)
    return avg_loss, acc

def train(train_loader, test_loader, epochs, optimizer, model, loss_fn, accuracy_fn,device):
    model.to(device)
    train_losses, test_losses, train_accuracies, test_accuracies = [], [], [], []
   
    for epoch in tqdm.trange(epochs, desc="Training"):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loss, train_acc = train_step(train_loader, model, optimizer, loss_fn, accuracy_fn, device)
        test_loss, test_acc = test_step(test_loader, model, loss_fn, accuracy_fn, device)
        train_losses.append(train_loss), train_accuracies.append(train_acc)
        test_losses.append(test_loss), test_accuracies.append(test_acc)

    return train_losses, test_losses, train_accuracies, test_accuracies


def train_workflow(model, train_dataset, val_dataset, config, device):
    epochs         = config.get('epochs', 5)
    lr             = config.get('lr', 1e-5)
    weight_decay   = config.get('weight_decay', 0.0)
    batch_size     = config.get('batch_size', 32)
    momentum       = config.get('momentum', 0.0)
    loss_fn        = config.get('loss_fn', torch.nn.CrossEntropyLoss(reduction='sum')) 

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False)
    
    train(train_loader, val_loader, epochs, optimizer, model, loss_fn, accuracy_fn, device)
    
    
    