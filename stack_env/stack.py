import warnings,logging
warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import Adam

import mlflow
from copy import deepcopy
from tqdm import tqdm
from itertools import chain

from networks import *
from utils import *
from configs import *


class main:
    def __init__(self, storage_path):
        self.hlp = HLP().to(hypers.device)  # high level policy
        self.qh1 = HL_Critic().to(hypers.device)
        self.qh2 = HL_Critic().to(hypers.device)
        self.qh1_target = deepcopy(self.qh1).to(hypers.device)
        self.qh2_target = deepcopy(self.qh2).to(hypers.device)
                
        self.llp = LLP().to(hypers.device) # low level policy
        self.ql1 = LL_Critic().to(hypers.device)
        self.ql2 = LL_Critic().to(hypers.device)
        self.ql1_target = deepcopy(self.ql1).to(hypers.device)
        self.ql2_target = deepcopy(self.ql2).to(hypers.device)
                
        self.hlp_optim = Adam(self.hlp.parameters(), lr=hypers.policy_lr)
        self.llp_optim = Adam(self.llp.parameters(), lr=hypers.policy_lr)
        self.qh_optim = Adam(chain(self.qh1.parameters(), self.qh2.parameters()), lr=hypers.critic_lr, fused=True)
        self.ql_optim = Adam(chain(self.ql1.parameters(), self.ql2.parameters()), lr=hypers.critic_lr, fused=True)

        #self.compile_()
        
        self.hl_entropy_target = -hypers.hl_action_dim
        self.hl_log_alpha = torch.tensor(1.0,requires_grad=True, device=hypers.device)  
        self.hl_alpha_optim = Adam([self.hl_log_alpha], lr=hypers.alpha_lr)

        self.ll_entropy_target = -hypers.ll_action_dim
        self.ll_log_alpha = torch.tensor(1.0,requires_grad=True, device=hypers.device)  
        self.ll_alpha_optim = Adam([self.ll_log_alpha], lr=hypers.alpha_lr)
        
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
        q_target = reward.squeeze() + hypers.gamma * (1-terminated) * (min_q_target-alpha.detach() * log_nx_actions.squeeze()) 
        return q_target
    
    def update_critics(self, q1_pred, q2_pred, q_target, optim):
        loss = F.smooth_l1_loss(q1_pred, q_target) + F.smooth_l1_loss(q2_pred, q_target) 
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step() 
        return loss
    
    def soft_update(self, q1, q2, q1_target, q2_target):
        for q1_pars,q1_target_pars in zip(q1.parameters(), q1_target.parameters()):
            q1_target_pars.data.mul_(1.0-hypers.tau).add_(q1_pars.data, alpha=hypers.tau)
                
        for q2_pars,q2_target_pars in zip(q2.parameters(), q2_target.parameters()):
            q2_target_pars.data.mul_(1.0-hypers.tau).add_(q2_pars.data, alpha=hypers.tau)

    def update_policy(self, q1, q2, alpha, log_pi, optim):
        min_q = torch.min(q1,q2)
        loss = ((alpha.detach()*log_pi.squeeze()) -  min_q).mean() 
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        return loss

    def tune_alpha(self, log_alpha, log_pi, entropy_target, optim):
        alpha_loss = (log_alpha * (-log_pi-entropy_target).detach()).mean()
        optim.zero_grad(set_to_none=True)
        alpha_loss.backward() 
        optim.step()
        alpha = log_alpha.exp()
        return alpha

    def train(self):
        mlflow.set_experiment("sac-stack-robosuite")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
    
            ll_alpha = self.ll_log_alpha.exp()
            hl_alpha = self.hl_log_alpha.exp()
            
            hlp_cpu = HLP().cpu().share_memory()   
            llp_cpu = LLP().cpu().share_memory()
            episodes_queue = mp.Queue(maxsize=hypers.queue_maxsize)
            p_list = []
            for n in range(hypers.num_rollout_workers):  # launching workers for data collection
                process = mp.Process(target=step_envs, args=(episodes_queue, hlp_cpu, llp_cpu,), daemon=True)
                p_list.append(process)
                process.start()

            buffer = create_buffer()
            for tensors in buffer: tensors.share_memory_()
            filler_worker = mp.Process(target=filler, args=(buffer, episodes_queue, run_id), daemon=True)
            filler_worker.start()

            gpu_stream = mp.Queue(maxsize=hypers.queue_maxsize)
            sampler_worker = mp.Process(target=sampler, args=(buffer, gpu_stream,), daemon=True)
            sampler_worker.start()
            
            print_queue_loading(gpu_stream, "gpu stream", 10)  # block here in a while loop while the gpu stream is getting filled 
            
            for c in tqdm(range(1000), total=1000):

                data = gpu_stream.get()
                data = [tensor.to(hypers.device, dtype=torch.float) for tensor in data]
                _states, _nx_states, _local_reward, _dones, _actions, _hl_goals, _obs_goals = data
                
                with torch.no_grad(): # -||s_t + g_t - s_t+1||_2 + gamma * (min(Q_(1,2)(st+1,gt+1,at+1)) - alpha * log pi(at+1|st+1,g_t+1))
                    nx_actions,log_nx_actions,_ = self.llp(_nx_states,_obs_goals)
                    q1 = self.ql1_target(_nx_states,nx_actions,_obs_goals)
                    q2 = self.ql2_target(_nx_states,nx_actions,_obs_goals)
                    q_target = self.compute_q_target(q1, q2, nx_actions, log_nx_actions, _nx_states, _obs_goals, _local_reward, _dones, ll_alpha)
                                     
                q1_pred = self.ql1(_states,_actions,_hl_goals).squeeze() 
                q2_pred = self.ql2(_states,_actions,_hl_goals).squeeze()
                ll_q_loss = self.update_critics(q1_pred, q2_pred, q_target, self.llp_optim) 

                self.soft_update(self.ql1, self.ql2, self.ql1_target, self.ql2_target) # q targets update
                
                new_action,log_pi,_ = self.llp(_states,_hl_goals)
                q1 = self.ql1(_states,new_action,_hl_goals).squeeze()
                q2 = self.ql2(_states,new_action,_hl_goals).squeeze()
                ll_policy_loss = self.update_policy(q1, q2, ll_alpha, log_pi, self.llp_optim) # alpla * log policy(at|st,g_t) - min(Q_(1,2)(st,g_t,at)) 

                llp_cpu.load_state_dict(self.llp.state_dict()) # update low level policy cpu state 

                ll_alpha = self.tune_alpha(self.ll_log_alpha, log_pi, self.ll_entropy_target, self.ll_alpha_optim) 

                for n in range(4):
                    data = gpu_stream.get()
                    data = [tensor.to(hypers.device, dtype=torch.float) for tensor in data]
                    _states, _nx_states, _local_reward, _dones, _actions, _hl_goals, _obs_goals = data
                    
                    # TODO : goal relabelling       

                    with torch.no_grad():  # env Reward + gamma * (min(Q_(1,2)(st+1,gt+1)) - alpha * log pi(at+1|st+1,g_t+1))
                        nx_actions,log_nx_actions,_ = self.hlp(_nx_states)
                        q1 = self.qh1_target(_nx_states,_obs_goals)
                        q2 = self.qh2_target(_nx_states,_obs_goals)
                        q_target = self.compute_q_target(q1, q2, nx_actions, log_nx_actions, _nx_states, _obs_goals, _reward, _dones, hl_alpha)
             
                    q1_pred = self.qh1(_states,_hl_goals).squeeze()
                    q2_pred = self.qh2(_states,_hl_goals).squeeze()
                    hl_q_loss = self.update_critics(q1_pred, q2_pred, q_target, self.hlp_optim) 

                    self.soft_update(self.qh1, self.qh2, self.qh1_target, self.qh2_target) # q targets update
                    
                    new_action,log_pi,_ = self.hlp(_states)
                    q1 = self.qh1(_states,new_action).squeeze()
                    q2 = self.qh2(_states,new_action).squeeze()
                    hl_policy_loss = self.update_policy(q1, q2, hl_alpha, log_pi, self.hlp_optim) # alpla * log policy(at|st) - min(Q_(1,2)(st,g_t)) 

                    hlp_cpu.load_state_dict(self.hlp.state_dict()) # update high level policy cpu state

                    hl_alpha = self.tune_alpha(self.hl_log_alpha, log_pi, self.hl_entropy_target, self.hl_alpha_optim)
        
                if c > 0 and c % int(20e3) == 0 :
                    self.n+=1 
                    self.save(self.n)
               

if __name__ == "__main__": 
    mp.set_start_method("spawn",force=True)
    mp.set_sharing_strategy("file_system")

    main(storage_path="./").train()
