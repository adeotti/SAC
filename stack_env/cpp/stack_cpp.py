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
  
    def train_low_level_policy(self, low_gpu_stream, llp_cpu, run_id):
        q1_net, q2_net = self.ql
        q1_target_net, q2_target_net = self.ql_target
        ll_alpha = self.ll_log_alpha.exp()

        low_stream = torch.cuda.Stream()
        
        for n in tqdm(range(hypers.max_llp_update_steps+1), total=hypers.max_llp_update_steps+1, desc="Low Level", position=0, leave=True):  
            with torch.cuda.stream(low_stream):
                _states, _nx_states, _local_reward, _dones, _actions, _hl_goals, _obs_goals = low_gpu_stream.get()
                pass

    def train_high_level_policy(self, high_gpu_stream, hlp_cpu, run_id):
        q1_net, q2_net = self.qh
        q1_target_net, q2_target_net = self.qh_target
        hl_alpha = self.hl_log_alpha.exp()

        high_stream = torch.cuda.Stream()
        for n in tqdm(range(hypers.max_hlp_update_steps+1), total=hypers.max_hlp_update_steps+1, desc="High Level", position=1, leave=True):
            with torch.cuda.stream(high_stream):
                _states, _nx_states, _reward, _dones, _actions, _hl_goals, _obs_goals = high_gpu_stream.get()
                pass
                 
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
