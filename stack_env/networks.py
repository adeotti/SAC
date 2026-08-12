import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


def weight_init(l):
    if isinstance(l,nn.Linear):
        nn.init.orthogonal_(l.weight)
        nn.init.constant_(l.bias,0.0)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l_mean = nn.Linear(256, hypers.action_dim)
        self.l_std = nn.Linear(256,hypers.action_dim)
        self.apply(weight_init)
        self.optim = torch.optim.Adam(self.parameters(),hypers.lr)

    def forward(self,obs):
        x = F.silu(self.l1(obs))
        x = F.silu(self.l2(x))
        
        mean = self.l_mean(x)
        log_std = self.l_std(x)
        log_std = torch.clamp(log_std,-20,2)
        std = log_std.exp()
        dist = Normal(mean,std) 
        
        pre_tanh = dist.rsample()
        action = F.tanh(pre_tanh)
        log = dist.log_prob(pre_tanh)
        log -=  2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh)) # torch.log(1-action.pow(2) + 1e-6) 
        log = log.sum(dim=-1,keepdim=True)  
        return action,log,torch.tanh(mean)
    

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim + hypers.action_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.output = nn.Linear(256,1)
        self.apply(weight_init) 

    def forward(self,obs,action): 
        cat = torch.cat((obs,action),dim=-1)
        x = F.silu(self.l1(cat))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        x = self.output(x)
        return x
