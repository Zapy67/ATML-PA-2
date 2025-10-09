import torch
from torchvision.models import resnet152, ResNet152_Weights
from torch import nn
from utils import unfreeze_layers

def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        x = torch.flatten(self.avgpool(f4), 1)
        x = self.fc(x)

        return x, [f3, f4]

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

            return 2 * mmd2  / len(self.sigmas_)
        
class DANLoss(nn.Module):
        def __init__(self, sigmas, scale):
            super().__init__()
            self.sigmas = sigmas
            self.scale = scale
            self.mkmmd = MKMMDLoss(sigmas)
            self.supervised = torch.nn.CrossEntropyLoss()
              
        def forward(self, source_features, target_features, logits, labels):
            loss = 0
            for H_s, H_t in zip(source_features, target_features):
                loss += self.scale*self.mkmmd(H_s, H_t)
            
            loss += self.supervised(logits, labels)     
            return loss   

dan_resnet = resnet152(ResNet152_Weights.DEFAULT)
dan_resnet.forward = _forward_impl

for p in dan_resnet.parameters():
     p.requires_grad = False

unfreeze_layers(dan_resnet, ['layer3','layer4', 'fc' ])