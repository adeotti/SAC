import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

__all__ = ["HL","HL_Critic","LL","LL_Critic"]

# observation dim = 162, action dim = 9 and goal dim = 6

def weight_init(l):
    if isinstance(l,nn.Linear):
        nn.init.orthogonal_(l.weight)
        nn.init.constant_(l.bias,0.0)


class HL(nn.Module):
    def __init__(self): # high level policy
        super().__init__()
        # target is cat[gripper_to_cubeA,cubeA_pos], shape 6 
        # to get the upper bound of the two values : gripper_to_cubeA.max() and cubeA_pos.max()  
        self.l1 = nn.Linear(162,256)
        self.lmean = nn.Linear(256,6)
        self.lstd = nn.Linear(256,6)
        self.apply(weight_init)

    def forward(self,obs):
        x = F.silu(self.l1(obs))

        mean = self.lmean(x)
        log_std = self.lstd(x)
        log_std = torch.clamp(log_std,-20,2)
        std = log_std.exp()
        dist = Normal(mean,std) 
        
        pre_tanh = dist.rsample()
        action = F.tanh(pre_tanh)
        log = dist.log_prob(pre_tanh)
        log -=  2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))  
        log = log.sum(dim=-1,keepdim=True)  
        return action,log,torch.tanh(mean)


class HL_Critic(nn.Module): # high level critic 
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(162+6,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.output = nn.Linear(256,1)
        self.apply(weight_init) 

    def forward(self,obs,goal): 
        cat = torch.cat((obs,goal),dim=-1)
        x = F.silu(self.l1(cat))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        return self.output(x)


class LL(nn.Module): # low level policy
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(162+6,256) # -> obs + goal -> 256
        self.l2 = nn.Linear(256,256)
        self.l_mean = nn.Linear(256,9)
        self.l_std = nn.Linear(256,9)
        self.apply(weight_init)
        
    def forward(self,obs,goal):
        x = torch.cat([obs,goal],dim=-1)
        x = F.silu(self.l1(x))
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
    

class LL_Critic(nn.Module): # low level critic 
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(162+9+6,256) # -> obs + action + goal -> 256
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.output = nn.Linear(256,1)
        self.apply(weight_init) 

    def forward(self,obs,action,goal): 
        cat = torch.cat((obs,action,goal),dim=-1)
        x = F.silu(self.l1(cat))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        return self.output(x)
