import warnings,logging
warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)

import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.optim import Adam

import mlflow
import time
import queue
from threading import Thread
from copy import deepcopy
from tqdm import tqdm

from networks import *
from utils import *
from configs import *

class main:
    def __init__(self, storage_path):        
        self.init_hlp()
        self.init_llp()
        self.storage_path = storage_path
        self.n = 0

    def init_hlp(self):
        self.hlp = HLP().to(hypers.device)  # high level policy
        self.hlp_optim = Adam(self.hlp.parameters(), lr=hypers.policy_lr, foreach=False)
        self.hlp.compile(mode="max-autotune")
    
        self.qh = nn.ModuleList([HL_Critic() for _ in range(2)]).to(hypers.device)
        self.qh_target = nn.ModuleList([deepcopy(net) for net in self.qh]).to(hypers.device)
        self.qh_optim = Adam(self.qh.parameters(), lr=hypers.critic_lr, fused=True)
        self.qh.compile(mode="max-autotune")

        self.hl_entropy_target = -hypers.hl_action_dim
        self.hl_log_alpha = torch.tensor(1.0,requires_grad=True, device=hypers.device)  
        self.hl_alpha_optim = Adam([self.hl_log_alpha], lr=hypers.alpha_lr)

    def init_llp(self):
        self.llp = LLP().to(hypers.device) # low level policy
        self.llp_optim = Adam(self.llp.parameters(), lr=hypers.policy_lr, foreach=False)
        self.llp.compile(mode="max-autotune")

        self.ql = nn.ModuleList([LL_Critic() for _ in range(2)]).to(hypers.device)
        self.ql_target = nn.ModuleList([deepcopy(net) for net in self.ql]).to(hypers.device)
        self.ql_optim = Adam(self.ql.parameters(), lr=hypers.critic_lr, fused=True)
        self.ql.compile(mode="max-autotune")

        self.ll_entropy_target = -hypers.ll_action_dim
        self.ll_log_alpha = torch.tensor(1.0,requires_grad=True, device=hypers.device)  
        self.ll_alpha_optim = Adam([self.ll_log_alpha], lr=hypers.alpha_lr)
            
    def save(self, step):
        check = {
            "high level state": self.hlp.state_dict(),
            "low level state": self.llp.state_dict(),             
        }
        torch.save(check,f"{self.storage_path}{step}.pth")
    
    def compute_q_target(self,q1, q2, log_nx_actions, reward, done, alpha, exp=1):
        assert q1.shape == q2.shape == log_nx_actions.shape == reward.shape == done.shape 
        min_q_target = torch.min(q1,q2)
        return reward + (hypers.gamma**exp) * (1 - done) * (min_q_target-alpha.detach() * log_nx_actions) # [1024, 1]  

    def get_critics_loss(self, q1_pred, q2_pred, q_target):    
        assert q1_pred.shape == q2_pred.shape == q_target.shape 
        return F.smooth_l1_loss(q1_pred, q_target) + F.smooth_l1_loss(q2_pred, q_target)
        
    def get_policy_loss(self, q1, q2, alpha, log_pi):
        assert q1.shape == q2.shape == log_pi.shape
        min_q = torch.min(q1,q2)
        return ((alpha.detach()*log_pi) -  min_q).mean(), min_q.squeeze() 
    
    def tune_alpha(self, log_alpha, log_pi, entropy_target, optim):
        alpha_loss = (log_alpha * (-log_pi-entropy_target).detach()).mean()
        optim.zero_grad()
        alpha_loss.backward() 
        optim.step()
        return log_alpha.exp()
    
    @torch.no_grad()
    def relabel_goals(self, _hl_goals, _states, _nx_states, _actions): 
        _states = _states.unsqueeze(2)
        sample_1 = _hl_goals.unsqueeze(1).unsqueeze(1)                                           # [1024, 1, 1, 6]
        sample_2 = (extract_goals(_nx_states.unsqueeze(1)) - extract_goals(_states[:, 0, :, :])) # [1024, 1, 6]
        sample_3 = Normal(loc=sample_2, scale=0.5).sample((8,)).permute(1, 2, 0, -1)             # [8, 1024, 6]     -->   [1024, 1, 8, 6]
        g_stack = torch.cat([sample_1, sample_2.unsqueeze(1), sample_3], dim=2)                  # [1024, 1, 10, 6]
        
        g_stack = g_stack.expand(-1, 10, -1, -1)                 # [1024,  1, 10,  6]  -->  [1024, 10, 10,  6]
        _actions = _actions.unsqueeze(2).expand(-1, -1, 10, -1)  # [1024, 10,  1,  9]  -->  [1024, 10, 10,  9]
        _states = _states.expand(-1, -1, 10, -1)                 # [1024, 10,  1, 81]  -->  [1024, 10, 10, 81]

        log = self.llp.evaluate_actions(_states, g_stack, _actions)  # [1024, 10, 10, 1]
        log = torch.sum(log, dim=1, keepdim=True)                        # [1024,  1, 10, 1]
        arg_max = torch.argmax(log, dim=2, keepdim=True)                 # [1024,  1,  1, 1] 
    
        idx = arg_max.expand(-1, 10, -1, hypers.hl_action_dim)                              # [1024, 10,  1, 6] 
        _hl_goals = torch.take_along_dim(g_stack, idx, dim=2).squeeze(2) # [1024, 10,  6]
        return _hl_goals

    def train_low_level_policy(self, low_gpu_stream, llp_cpu, run_id):
        q1_net, q2_net = self.ql
        q1_target_net, q2_target_net = self.ql_target
        ll_alpha = self.ll_log_alpha.exp()

        low_stream = torch.cuda.Stream()
        
        for n in tqdm(range(hypers.max_llp_update_steps+1), total=hypers.max_llp_update_steps+1, desc="Low Level", position=0, leave=True):  
            with torch.cuda.stream(low_stream):
                _states, _nx_states, _local_reward, _dones, _actions, _hl_goals, _obs_goals = low_gpu_stream.get()

                with torch.no_grad():  
                    nx_actions, log_nx_actions,_ = self.llp(_nx_states, _hl_goals)
                    q1 = q1_target_net(_nx_states, nx_actions, _hl_goals)
                    q2 = q2_target_net(_nx_states, nx_actions, _hl_goals)
                    q_target = self.compute_q_target(q1, q2, log_nx_actions, _local_reward, _dones, ll_alpha)
                                         
                q1_pred = q1_net(_states, _actions, _hl_goals) 
                q2_pred = q2_net(_states, _actions, _hl_goals)
                ll_q_loss = self.get_critics_loss(q1_pred, q2_pred, q_target) 
                self.ql_optim.zero_grad()
                ll_q_loss.backward()
                self.ql_optim.step()
                                    
                new_action,log_pi,_ = self.llp(_states,_hl_goals)
                q1 = q1_net(_states, new_action, _hl_goals)
                q2 = q2_net(_states, new_action, _hl_goals)
                ll_policy_loss, min_q = self.get_policy_loss(q1, q2, ll_alpha, log_pi)  
                self.llp_optim.zero_grad()
                ll_policy_loss.backward()
                self.llp_optim.step()

                with torch.no_grad():  # update target nets and cpu weights
                    torch._foreach_lerp_(list(self.ql_target.parameters()), list(self.ql.parameters()), hypers.tau)

                    for gpu_params, cpu_params in zip(self.llp.parameters(), llp_cpu.parameters()):
                        cpu_params.copy_(gpu_params)
        
                ll_alpha = self.tune_alpha(self.ll_log_alpha, log_pi, self.ll_entropy_target, self.ll_alpha_optim)
                
                if n % int(1e4) == 0:
                    low_stream.synchronize()

                    mlflow.log_metrics(
                        {   
                            "Low Level/low level critic min q": min_q.mean().item(),
                            "Low Level/low level critic loss": ll_q_loss.item(),
                            "Low Level/low level policy loss": ll_policy_loss.item(),
                            "Low Level/low level alpha": ll_alpha.item(),
                            "Low Level/low reward": _local_reward.mean().item(),
                        },
                        step = n, run_id = run_id
                    )

                    if n % int(2e5) == 0:
                        self.n +=1
                        self.save(self.n)

    def train_high_level_policy(self, high_gpu_stream, hlp_cpu, run_id):
        q1_net, q2_net = self.qh
        q1_target_net, q2_target_net = self.qh_target
        hl_alpha = self.hl_log_alpha.exp()

        high_stream = torch.cuda.Stream()
        for n in tqdm(range(hypers.max_hlp_update_steps+1), total=hypers.max_hlp_update_steps+1, desc="High Level", position=1, leave=True):
            with torch.cuda.stream(high_stream):
                _states, _nx_states, _reward, _dones, _actions, _hl_goals, _obs_goals = high_gpu_stream.get()
                
                _hl_goals = self.relabel_goals(_hl_goals, _states, _nx_states, _actions)
                _states = _states[:,0,:]
                _hl_goals = _hl_goals[:,0,:]
                
                with torch.no_grad():
                    nx_actions, log_nx_actions,_ = self.hlp(_nx_states)
                    q1 = q1_target_net(_nx_states, nx_actions)
                    q2 = q2_target_net(_nx_states, nx_actions)
                    q_target = self.compute_q_target(q1, q2, log_nx_actions, _reward, _dones, hl_alpha, exp=10)
             
                q1_pred = q1_net(_states, _hl_goals)
                q2_pred = q2_net(_states, _hl_goals)
                hl_q_loss = self.get_critics_loss(q1_pred, q2_pred, q_target) 
                self.qh.zero_grad()
                hl_q_loss.backward()
                self.qh_optim.step()

                new_action,log_pi,_ = self.hlp(_states)
                q1 = q1_net(_states, new_action)
                q2 = q2_net(_states, new_action)
                hl_policy_loss, min_q = self.get_policy_loss(q1, q2, hl_alpha, log_pi)  
                self.hlp_optim.zero_grad()
                hl_policy_loss.backward()
                self.hlp_optim.step()
                
                with torch.no_grad(): # update target nets and cpu weights
                    torch._foreach_lerp_(list(self.qh_target.parameters()), list(self.qh.parameters()), hypers.tau)

                    for gpu_params, cpu_params in zip(self.hlp.parameters(), hlp_cpu.parameters()):
                        cpu_params.copy_(gpu_params)

                hl_alpha = self.tune_alpha(self.hl_log_alpha, log_pi, self.hl_entropy_target, self.hl_alpha_optim)
                
                if  n % int(1e4) == 0:
                    high_stream.synchronize()

                    mlflow.log_metrics(
                        {   
                            "High Level/high level critic min q": min_q.mean().item(),
                            "High Level/high level critic loss": hl_q_loss.item(),
                            "High Level/high level policy loss": hl_policy_loss.item(),
                            "High Level/high level alpha": hl_alpha.item(),
                            "High Level/high reward": _reward.mean().item()
                        },
                        step = n, run_id = run_id
                    ) 


    def train(self):
        mlflow.set_experiment("sac-stack-robosuite")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
                      
            hlp_cpu = HLP().cpu().share_memory()   
            llp_cpu = LLP().cpu().share_memory()
            
            try:
                episodes_queue = mp.Queue(maxsize=hypers.low_queue_maxsize)
                processes_list = []
                events = [mp.Event() for p in range(hypers.num_rollout_workers)]
                for n in range(hypers.num_rollout_workers):  # launching workers for data collection
                    process = mp.Process(target=step_envs, args=(episodes_queue, hlp_cpu, llp_cpu, events[n]), daemon=True)
                    processes_list.append(process)
                    process.start()

                buffer = create_buffer()
                filler_worker = mp.Process(target=filler, args=(buffer, episodes_queue, run_id))
                filler_worker.start()

                print("warmup start")
                while not buffer[-1].item() >= hypers.buffer_min_capacity: # waiting unting the current buffer capacity is sufficient
                    time.sleep(0.2)

                for flag in events[4:]:
                    flag.set()

                for process in processes_list[4:]:
                    process.join()
      
                low_gpu_stream = queue.Queue(maxsize=hypers.low_queue_maxsize)  # low level stream 
                low_sampler_worker = Thread(target=low_level_sampler, args=(buffer, low_gpu_stream,), daemon=True)
                low_sampler_worker.start()
                
                high_gpu_stream = queue.Queue(maxsize=hypers.high_queue_maxsize)  # high level stream 
                high_sampler_worker = Thread(target=high_level_sampler, args=(buffer, high_gpu_stream), daemon=True)
                high_sampler_worker.start()
                
                while not low_gpu_stream.full():
                    time.sleep(0.2)

                while not high_gpu_stream.full():
                    time.sleep(0.2)
                
                print("launching training workers")
                low_worker = Thread(target=self.train_low_level_policy, args=(low_gpu_stream, llp_cpu, run_id), daemon=True)
                high_worker = Thread(target=self.train_high_level_policy, args=(high_gpu_stream, hlp_cpu, run_id), daemon=True)
                low_worker.start() ; high_worker.start()
                low_worker.join()  ; high_worker.join()

            except Exception as error:
                print(error)

            finally:
                for process in processes_list: process.terminate() 
                filler_worker.terminate()


if __name__ == "__main__": 
    mp.set_start_method("spawn", force=True)
    mp.set_sharing_strategy("file_system") 

    main(storage_path="./").train()
