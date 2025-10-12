import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_features(backbone, dataset, batchsize=32, device='cpu'):
    features = []
    
    backbone.eval() 
    backbone.to(device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=False)
    labels = []

    with torch.inference_mode():
        for i, (X,Y) in enumerate(loader):
            X, Y = X.to(device), Y.cpu()
            feats = backbone(X)
            labels.append(Y)
            features.append(feats)
            if i%5 == 0:
                print(f"progress : {len(labels)}|{len(loader)}")
        
        print(f"progress : {len(labels)}|{len(loader)}")
        
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def tsne_plot(latents, labels, classes=None, ax=None, fig=None, num_samples=5000, perplexity=30, random_state=500):
    if latents.shape[0] > num_samples:
        idx = torch.randperm(latents.shape[0])[:num_samples]
        latents = latents[idx]
        labels = labels[idx]
 
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state)
    z_embed = tsne.fit_transform(latents)
 
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(z_embed[:, 0], z_embed[:, 1], c=labels, cmap=[str(l) for l in labels], s=5, alpha=0.7)

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(len(classes) if classes else len(set(labels))))
    if classes:
        cbar.ax.set_yticklabels(classes)
    cbar.set_label("Classes")

    return fig, ax