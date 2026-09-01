import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from configs import hypers

__all__ = ["HLP", "HL_Critic", "LLP", "LL_Critic"]

# observation dim = 81, action dim = 9 and goal dim = 6

def weight_init(l):
    if isinstance(l,nn.Linear):
        nn.init.orthogonal_(l.weight)
        nn.init.constant_(l.bias,0.0)


class HLP(nn.Module):
    def __init__(self): # high level policy, target is cat[gripper_to_cubeA,cubeA_pos], shape 6 
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim, 256)
        self.lmean = nn.Linear(256, hypers.hl_action_dim)
        self.lstd = nn.Linear(256, hypers.hl_action_dim)
        self.apply(weight_init)

    def forward(self, obs):
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

        scaled_mean = torch.tanh(mean) #* 5.0
        return action, log, scaled_mean


class HL_Critic(nn.Module): # high level critic 
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim + hypers.hl_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        self.apply(weight_init) 

    def forward(self,obs,goal): 
        cat = torch.cat((obs,goal), dim=-1)
        x = F.silu(self.l1(cat))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        return self.output(x)


class LLP(nn.Module): # low level policy
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim + hypers.hl_action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l_mean = nn.Linear(256, hypers.ll_action_dim)
        self.l_std = nn.Linear(256, hypers.ll_action_dim)
        self.apply(weight_init)

    def get_dist(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        x = F.silu(self.l1(x))
        x = F.silu(self.l2(x))
        
        mean = self.l_mean(x)
        log_std = torch.clamp(self.l_std(x), -20, 2)
        std = log_std.exp()
        return Normal(mean, std)

    def forward(self, obs, goal):
        dist = self.get_dist(obs, goal)
        pre_tanh = dist.rsample()
        
        log_prob = dist.log_prob(pre_tanh)
        log_prob = self.reparam(log_prob, pre_tanh)
        
        action = torch.tanh(pre_tanh)
        mean_action = torch.tanh(dist.mean)
        return action, log_prob, mean_action

    def evaluate_actions(self, obs, goal, actions):
        dist = self.get_dist(obs, goal)
        clamped_actions = torch.clamp(actions, -0.999999, 0.999999)
        pre_tanh = torch.atanh(clamped_actions)
        log_prob = dist.log_prob(pre_tanh)
        return self.reparam(log_prob, pre_tanh)

    def reparam(self, log, pre_tanh):
        log -= 2 * (np.log(2) - pre_tanh - F.softplus(-2 * pre_tanh))
        log = log.sum(dim=-1, keepdim=True)
        return log


class LL_Critic(nn.Module): # low level critic 
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim + hypers.ll_action_dim + hypers.hl_action_dim, 256) # -> obs + action + goal -> 256
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 1)
        self.apply(weight_init) 

    def forward(self,obs,action,goal): 
        cat = torch.cat((obs,action,goal), dim=-1)
        x = F.silu(self.l1(cat))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        return self.output(x)
