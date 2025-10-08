import torch
import argparse
import os
import matplotlib.pyplot as plt

from architecture import dan_resnet, DANLoss
from utils import train
import dataset

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
    
    source_loader = dataset.source_loader
    target_loader = dataset.target_loader
    
    loss_fn = DANLoss(sigmas=[1.0, 5.0, 10.0], scale=1.0)
    optimizer = torch.optim.SGD(dan_resnet.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    
    source_losses, supervised_losses, mkmmd_losses, train_accs, test_accs = train(
        source_loader, target_loader, target_loader,
        epochs=args.epochs, optimizer=optimizer, model=dan_resnet,
        loss_fn=loss_fn, accuracy_fn=args.accuracy_fn, device=device
    )

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

    os.mkdir('figs', exist_ok=True)
    plt.savefig('figs/training_dynamics.png')
    torch.save(dan_resnet.state_dict(), args.save_path)
