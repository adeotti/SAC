import warnings,logging
warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)

import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.common import Autoreset

import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

import sys
import mlflow
import queue
from copy import deepcopy
from tqdm import tqdm
from itertools import chain
from threading import Thread

from networks import *
from configs import hypers,env_configs


def vec_env():
    def make_env():
        x = robosuite.make(env_name="Stack", **env_configs) 
        x = GymWrapper(x,list(x.observation_spec()))
        x.metadata = {"render_mode":[]}
        x = Autoreset(x)
        return x 
    env = SyncVectorEnv([make_env for _ in range(hypers.num_envs)])
    return env


def get_local_reward(env,hl_goal): # TODO : review math
    mujoco_layer = env.unwrapped
    data_list = []
    for env in mujoco_layer.envs:
        cubeA_pos = env.unwrapped._get_observations()["cubeA_pos"]
        gripper_to_cubeA = env.unwrapped._get_observations()["gripper_to_cubeA"]
        stack = torch.cat([torch.from_numpy(cubeA_pos),torch.from_numpy(gripper_to_cubeA)])
        data_list.append(stack) 

    obs_goal = torch.stack(data_list,dim=0).float() # each env data are on the x axis
    diff = torch.linalg.norm((hl_goal-obs_goal),dim=-1)
    return diff.mean(), obs_goal


def create_storage(): # storage for the step env method where episodes steps are stored
    obs_dim = (hypers.num_envs,hypers.obs_dim)     
    act_dim = (hypers.num_envs,hypers.ll_action_dim)
    return (
        torch.empty((hypers.horizon, *obs_dim), dtype=torch.half), # states
        torch.empty((hypers.horizon, *obs_dim), dtype=torch.half), # nx states
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.half), # reward
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.half), # local reward 
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.bool), # done
        torch.empty((hypers.horizon, *act_dim), dtype=torch.half), # actions
        torch.empty((hypers.horizon, hypers.num_envs, 6), dtype=torch.half), # hl goal 
        torch.empty((hypers.horizon, hypers.num_envs, 6), dtype=torch.half) # observed goal
    )


class main:
    def __init__(self,storage_path):
        self.env = vec_env()

        self.hlp = HL().to(hypers.device)  # high level policy
        self.qh1 = HL_Critic().to(hypers.device)
        self.qh2 = HL_Critic().to(hypers.device)
        self.qh1_target = deepcopy(self.qh1).to(hypers.device)
        self.qh2_target = deepcopy(self.qh2).to(hypers.device)
                
        self.llp = LL().to(hypers.device) # low level policy
        self.ql1 = LL_Critic().to(hypers.device)
        self.ql2 = LL_Critic().to(hypers.device)
        self.ql1_target = deepcopy(self.ql1).to(hypers.device)
        self.ql2_target = deepcopy(self.ql2).to(hypers.device)
                
        self.hlp_optim = Adam(self.hlp.parameters(),lr=0.0001)
        self.llp_optim = Adam(self.llp.parameters(),lr=0.0001)
        self.qh_optim = Adam(chain(self.qh1.parameters(),self.qh2.parameters()),lr=0.001,fused=True)
        self.ql_optim = Adam(chain(self.ql1.parameters(),self.ql2.parameters()),lr=0.001,fused=True)

        #self.compile_()
        
        self.hl_entropy_target = -hypers.hl_action_dim
        self.hl_log_alpha = torch.tensor(1.0,requires_grad=True,device=hypers.device)  
        self.hl_alpha_optim = Adam([self.hl_log_alpha],lr=hypers.lr)

        self.ll_entropy_target = -hypers.ll_action_dim
        self.ll_log_alpha = torch.tensor(1.0,requires_grad=True,device=hypers.device)  
        self.ll_alpha_optim = Adam([self.ll_log_alpha],lr=hypers.lr)
        
        self.storage_path = storage_path
        self.n = 0 # tracking number for model data saving  

    def compile_(self):
        self.hlp.compile(mode="max-autotune")
        self.qh1.compile()
        self.qh2.compile()
        self.qh1_target.compile()
        self.qh2_target.compile()
        
        self.actor.compile(mode="max-autotune")
        self.ql1.compile()
        self.ql2.compile()
        self.ql1_target.compile()
        self.ql2_target.compile()
            
    def save(self, step):
        check = {
            "high level state": self.hlp.state_dict(),
            "low level state": self.actor.state_dict(),             
        }
        torch.save(check,f"{self.storage_path}{step}.pth")

    def compute_q_target(self,q1, q2, nx_actions, log_nx_actions, nx_states, obs_goals, reward, terminated, alpha):
        min_q_target = torch.min(q1,q2).squeeze()
        q_target = reward.squeeze() + hypers.gamma * (1-terminated) * (min_q_target - alpha.detach() * log_nx_actions.squeeze()) 
        return q_target
    
    def update_critics(self, q1_pred, q2_pred, q_target, optim):
        loss = F.smooth_l1_loss(q1_pred, q_target) + F.smooth_l1_loss(q2_pred, q_target) 
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step() 
        return loss
    
    def soft_update(self, q1, q2, q1_target, q2_target):
        for q1_pars,q1_target_pars in zip(q1.parameters(), q1_target.parameters()):
            q1_target_pars.data.mul_(1.0 - hypers.tau).add_(q1_pars.data,alpha=hypers.tau)
                
        for q2_pars,q2_target_pars in zip(q2.parameters(), q2_target.parameters()):
            q2_target_pars.data.mul_(1.0 - hypers.tau).add_(q2_pars.data,alpha=hypers.tau)

    def update_policy(self, q1, q2, alpha, log_pi, optim):
        min_q = torch.min(q1,q2)
        loss = ((alpha.detach()*log_pi.squeeze()) -  min_q).mean() 
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        return loss

    def tune_alpha(self, log_alpha, log_pi, entropy_target, optim):
        alpha_loss = (log_alpha * (-log_pi - entropy_target).detach()).mean()
        optim.zero_grad(set_to_none=True)
        alpha_loss.backward() 
        optim.step()
        alpha = log_alpha.exp()
        return alpha

    def train(self):
        mlflow.set_experiment("sac-stack-robosuite")
        with mlflow.start_run() as run:
    
            ll_alpha = self.ll_log_alpha.exp()
            hl_alpha = self.hl_log_alpha.exp()
            state = torch.from_numpy(self.env.reset()[0]).to(hypers.device,dtype=torch.float)

            for c in tqdm(range(1000),total=1000):
                hl_goal,_,_ = self.hlp(state) # sample goal 
                    
                stor_curr_states, stor_nx_states, stor_rewards, stor_local_reward, \
                stor_terminated, stor_actions, stor_hl_goal, stor_obs_goal = create_storage()     

                pointer = 0
                global_step = 0  
                for n in range(50): # 500
                    with torch.no_grad():
                        if global_step < hypers.warmup:  
                            action = self.env.action_space.sample()
                        else:
                            action,_,_ = self.llp(torch.as_tensor(state),hl_goal)
                            action = action.squeeze()
                        
                        nx_state,env_reward,done,trunc,info = self.env.step(action.tolist())
                        reward,obs_goal = get_local_reward(self.env, hl_goal.cpu())
                        reward -= reward
                        
                        saved_action = (torch.from_numpy(np.array(action)) if isinstance(action,np.ndarray) else action)
                        
                        stor_curr_states[pointer].copy_(torch.as_tensor(state))
                        stor_nx_states[pointer].copy_(torch.as_tensor(nx_state))
                        stor_rewards[pointer].copy_(torch.from_numpy(env_reward))
                        stor_local_reward[pointer].copy_(reward)
                        stor_terminated[pointer].copy_(torch.from_numpy(done))
                        stor_actions[pointer].copy_(saved_action)
                        stor_hl_goal[pointer].copy_(hl_goal)
                        stor_obs_goal[pointer].copy_(obs_goal)

                        obs = nx_state
                        pointer+=1
                        global_step += 1

                for n in range(50):
                    idx = torch.randperm(50)
                    _states = stor_curr_states[idx].to(hypers.device,dtype=torch.float)
                    _nx_states = stor_nx_states[idx].to(hypers.device,dtype=torch.float)
                    _local_reward = stor_local_reward[idx].to(hypers.device,dtype=torch.float).squeeze()
                    _terminated = stor_terminated[idx].to(hypers.device,dtype=torch.float)
                    _actions = stor_actions[idx].to(hypers.device,dtype=torch.float)
                    _hl_goals = stor_hl_goal[idx].to(hypers.device,dtype=torch.float)
                    _obs_goals = stor_obs_goal[idx].to(hypers.device,dtype=torch.float)
                    
                    with torch.no_grad(): # -||s_t + g_t - s_t+1||_2 + gamma * (min(Q_(1,2)(st+1,gt+1,at+1)) - alpha * log pi(at+1|st+1,g_t+1))
                        nx_actions,log_nx_actions,_ = self.llp(_nx_states,_obs_goals)
                        q1 = self.ql1_target(_nx_states,nx_actions,_obs_goals)
                        q2 = self.ql2_target(_nx_states,nx_actions,_obs_goals)
                        q_target = self.compute_q_target(q1, q2, nx_actions, log_nx_actions, _nx_states, _obs_goals, _local_reward, _terminated, ll_alpha)
                                         
                    q1_pred = self.ql1(_states,_actions,_hl_goals).squeeze() 
                    q2_pred = self.ql2(_states,_actions,_hl_goals).squeeze()
                    ll_q_loss = self.update_critics(q1_pred, q2_pred, q_target, self.llp_optim) 

                    self.soft_update(self.ql1, self.ql2, self.ql1_target, self.ql2_target) # q targets update
                    
                    new_action,log_pi,_ = self.llp(_states,_hl_goals)
                    q1 = self.ql1(_states,new_action,_hl_goals).squeeze()
                    q2 = self.ql2(_states,new_action,_hl_goals).squeeze()
                    ll_policy_loss = self.update_policy(q1, q2, ll_alpha, log_pi, self.llp_optim) # alpla * log policy(at|st,g_t) - min(Q_(1,2)(st,g_t,at))

                    ll_alpha = self.tune_alpha(self.ll_log_alpha, log_pi, self.ll_entropy_target, self.ll_alpha_optim) 

                for n in range(5):
                    idx = torch.randperm(50)
                    _states = stor_curr_states[idx].to(hypers.device,dtype=torch.float)
                    _nx_states = stor_nx_states[idx].to(hypers.device,dtype=torch.float)
                    _reward = stor_rewards[idx].to(hypers.device,dtype=torch.float).squeeze() # env reward
                    _terminated = stor_terminated[idx].to(hypers.device,dtype=torch.float)
                    _actions = stor_actions[idx].to(hypers.device,dtype=torch.float)
                    _hl_goals = stor_hl_goal[idx].to(hypers.device,dtype=torch.float)
                    _obs_goals = stor_obs_goal[idx].to(hypers.device,dtype=torch.float)
                    
                    # TODO : goal relabelling       

                    with torch.no_grad():
                        nx_actions,log_nx_actions,_ = self.hlp(_nx_states)
                        q1 = self.qh1_target(_nx_states,_obs_goals)
                        q2 = self.qh2_target(_nx_states,_obs_goals)
                        q_target = self.compute_q_target(q1, q2, nx_actions, log_nx_actions, _nx_states, _obs_goals, _reward, _terminated, hl_alpha)
             
                    q1_pred = self.qh1(_states,_hl_goals).squeeze()
                    q2_pred = self.qh2(_states,_hl_goals).squeeze()
                    hl_q_loss = self.update_critics(q1_pred, q2_pred, q_target, self.hlp_optim) 

                    self.soft_update(self.qh1, self.qh2, self.qh1_target, self.qh2_target) # q targets update
                    
                    new_action,log_pi,_ = self.hlp(_states)
                    q1 = self.qh1(_states,new_action).squeeze()
                    q2 = self.qh2(_states,new_action).squeeze()
                    hl_policy_loss = self.update_policy(q1, q2, hl_alpha, log_pi, self.hlp_optim) 

                    hl_alpha = self.tune_alpha(self.hl_log_alpha, log_pi, self.hl_entropy_target, self.hl_alpha_optim)

                #state = None # TODO update new state from data 
        
                if c > 0 and c % int(20e3) == 0 :
                    self.n+=1 ; self.save(self.n)
               

if __name__ == "__main__": 
    main(storage_path="./").train()
