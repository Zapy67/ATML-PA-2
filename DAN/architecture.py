import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

def resnet_classifier(num_classes, device='cpu'):
    classifier = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, num_classes)
    classifier.to(device)
    return classifier

class GaussianKernel(nn.Module):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma

    def forward(self, x, y):
        diff = x - y
        dist = torch.sum(diff ** 2, dim= tuple(range(1, x.dim())))
        return torch.exp(-dist / (2 * self.sigma ** 2))

class MKMMDLoss(nn.Module):        
        def __init__(self, sigmas_):
            super().__init__()
            self.sigmas_ = sigmas_
              
        def forward(self, H_s, H_t):
            assert len(H_s) == len(H_t)
            assert len(H_s) % 2 == 0

            half_size = len(H_s) // 2

            xs1, xs2 = H_s[0:half_size], H_s[half_size:]
            xt1, xt2 = H_t[0:half_size], H_t[half_size:]

            mmd2 = 0
            for sigma in self.sigmas_:
                k = GaussianKernel(sigma)
                k_ss = k(xs1, xs2)
                k_tt = k(xt1, xt2)
                k_st = k(xs1, xt2)
                k_ts = k(xs2, xt1)
                mmd2 += (k_ss + k_tt - k_st - k_ts).mean()
            
            return torch.clamp( 2* mmd2  / len(self.sigmas_), min=0)
        
class DANLoss(nn.Module):
        def __init__(self, sigmas, scale):
            super().__init__()
            self.sigmas = sigmas
            self.scale = scale
            self.mkmmd = MKMMDLoss(sigmas)
            self.supervised = torch.nn.CrossEntropyLoss(reduction='sum')
              
        def forward(self, source_features, target_features, logits, labels):
            scaled_mkmmd = 0
            for H_s, H_t in zip(source_features, target_features):
                scaled_mkmmd += self.scale*self.mkmmd(H_s, H_t)
            
            supervised = self.supervised(logits, labels) 
            
            return supervised, scaled_mkmmd   
            
