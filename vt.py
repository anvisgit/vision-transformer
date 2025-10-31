import PIL
import time
import torch
import torchvision
import torch.nn.functional as tnf
from einops import rearrange #reloading, reshaping tensor dimentions
from torch import nn
class residual(nn.Module):
    def __init__(self,layer):
        super().__init__()
        self.layer=layer
    def forward(self,x,**kwargs):
        return self.layer(x,**kwargs)+x #resnet
class normalize(nn.Module):
    def __init__(self,dim, layer):
        super().__init__()
        self.n=nn.LayerNorm(dim)
        self.layer=layer
    def forward(self,x,**kwargs):
        return self.layer(self.n(x), **kwargs)
class feedforward(nn.Module):
    def __init__(self,dim,diminit,dropout=0.1):
        super().__init__()
        self.nn1=nn.Linear(dim,diminit)
        torch.nn.init.xavier_uniform_(self.nn1.weight)
        torch.nn.init.normal_(self.nn1.bais,std=1e-6) #gives bias minute deviation
        self.af1=nn.GELU()
        self.do1=nn.Dropout(dropout)

        self.nn2=nn.Linear(diminit, dim)
        torch.nn.init.xavier_uniform_(self.nn2.weight)
        torch.nn.init.normal_(self.nn2.bias,std=1e-6)
        self.do2=nn.Dropout(dropout)

    def forward(self,x):
        x=self.nn1(x)
        x=self.af1(x)
        x=self.do1(x)
        x=self.nn2(x)
        x=self.do2(x)

        return x 
class Attention(nn.Module):
    def __init__(self,dim,head=10,dropout=0.1):
        self.dim=dim
        self.normfactor=dim**-0.5

        