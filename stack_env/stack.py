import warnings,logging
warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)

import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.common import Autoreset

import torch
from torch.optim import Adam
import torch.multiprocessing as mp
import numpy as np

import sys,time,mlflow,queue
from copy import deepcopy
from tqdm import tqdm
from itertools import chain
from threading import Thread

from goals import *
from utils import *
from networks import HL,LL,Critic
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


def create_storage(): # storage for the step env method where episodes steps are stored
    obs_dim = (hypers.num_envs,hypers.obs_dim)     
    act_dim = (hypers.num_envs,hypers.action_dim)
    return (
        torch.empty((hypers.horizon,*obs_dim),dtype=torch.half),
        torch.empty((hypers.horizon,*obs_dim),dtype=torch.half),
        torch.empty((hypers.horizon,hypers.num_envs,),dtype=torch.half),
        torch.empty((hypers.horizon,hypers.num_envs,),dtype=torch.bool),
        torch.empty((hypers.horizon,*act_dim),dtype=torch.half) 
    )


def step(queue,policy,mean,var,count,stat_logger=False): # main method for stepping in the envs and collecting transitions 
    with torch.no_grad():
        env = vec_env()
        stor_curr_states,stor_nx_states,stor_rewards,stor_terminated,stor_actions = create_storage()     
        pointer = 0
        global_step = 0
        obs = torch.from_numpy(env.reset()[0])

        while True:
            if global_step < hypers.warmup:
                action = env.action_space.sample()
            else:
                action,_,_ = policy(torch.as_tensor(obs))
                action = action.squeeze()
             
            nx_state,reward,done,trunc,info = env.step(action.tolist())
            
            saved_action = (torch.from_numpy(np.array(action)) if isinstance(action,np.ndarray) else action)
            
            stor_curr_states[pointer].copy_(torch.as_tensor(obs))
            stor_nx_states[pointer].copy_(torch.as_tensor(nx_state))
            stor_rewards[pointer].copy_(torch.from_numpy(reward))
            stor_terminated[pointer].copy_(torch.from_numpy(done))
            stor_actions[pointer].copy_(saved_action)

            obs = nx_state
            pointer+=1
            global_step += 1

            if pointer == hypers.horizon:
                data = (stor_curr_states.clone(),stor_nx_states.clone(),stor_rewards.clone(),stor_terminated.clone(),stor_actions.clone())
                queue.put(data)
                pointer = 0 
                stor_curr_states,stor_nx_states,stor_rewards,stor_terminated,stor_actions = create_storage()
        
        env.close()


def create_buffer(): # buffer storage where many episode are stored for random sampling later
    n_batch = torch.tensor(100) 
    b_state = torch.zeros((n_batch,hypers.horizon,hypers.num_envs,hypers.obs_dim),dtype=torch.half)
    b_nx_state = torch.zeros(n_batch,hypers.horizon,hypers.num_envs,hypers.obs_dim,dtype=torch.half)
    b_rewards = torch.zeros(n_batch,hypers.horizon,hypers.num_envs,dtype=torch.half)
    b_terminated = torch.zeros(n_batch,hypers.horizon,hypers.num_envs,dtype=torch.bool)
    b_actions = torch.zeros(n_batch,hypers.horizon,hypers.num_envs,hypers.action_dim,dtype=torch.half)
    current_capacity = torch.tensor(0)
    return (n_batch,b_state,b_nx_state,b_rewards,b_terminated,b_actions,current_capacity)


def filler(buffer,ep_queue,mlflow_run_id): # method for filling the buffer
    n_batch,b_state,b_nx_state,b_rewards,b_terminated,b_actions,current_capacity = buffer
    global_idx = 0

    while True: 
        ep_curr_state,ep_nx_state,ep_rewards,ep_terminated,ep_actions = ep_queue.get()
            
        p = global_idx % n_batch.item()

        b_state[p].copy_(ep_curr_state)
        b_nx_state[p].copy_(ep_nx_state)
        b_rewards[p].copy_(ep_rewards)
        b_terminated[p].copy_(ep_terminated)
        b_actions[p].copy_(ep_actions)
        
        global_idx += 1
        current_capacity.copy_(torch.tensor(min(global_idx, n_batch.item())))
        
        mean_return = ep_rewards.sum().item() / hypers.num_envs # tracking rewards per episodes
        mlflow.log_metric("Main/mean reward",mean_return,run_id=mlflow_run_id,step=global_idx) 
        

def sampler(buffer,gpu_stream): # method for sampling from the buffer
    n_batch,b_state,b_nx_state,b_rewards,b_terminated,b_actions,current_capacity = buffer
 
    while True:
        if current_capacity.item() < 20:
            time.sleep(0.1)
            continue
        
        idx_chunks = torch.randint(0,current_capacity,(hypers.batch_size,))
        idx_horizons = torch.randint(0,hypers.horizon,(hypers.batch_size,))
        idx_envs = torch.randint(0,hypers.num_envs,(hypers.batch_size,))

        s_states = b_state[idx_chunks,idx_horizons,idx_envs]
        s_nx_state = b_nx_state[idx_chunks,idx_horizons,idx_envs]
        s_reward = b_rewards[idx_chunks,idx_horizons,idx_envs].unsqueeze(-1)
        s_terminated = b_terminated[idx_chunks,idx_horizons,idx_envs].unsqueeze(-1)
        s_actions = b_actions[idx_chunks,idx_horizons,idx_envs]
        
        try:
            gpu_stream.put((s_states,s_nx_state,s_reward,s_terminated,s_actions),timeout=1.0)
        except queue.Full:
            continue


class main:
    def __init__(self,storage_path):
        self.hlp = HL().to(hypers.device)  # high level policy
        
        self.actor = LL().to(hypers.device) # low level policy
        self.q1 = Critic().to(hypers.device)
        self.q2 = Critic().to(hypers.device)

        self.q1_target = deepcopy(self.q1).to(hypers.device)
        self.q2_target = deepcopy(self.q2).to(hypers.device)
        
        self.hlp.compile(mode="max-autotune")
        self.actor.compile(mode="max-autotune")
        self.q1.compile()
        self.q2.compile()
        self.q1_target.compile()
        self.q2_target.compile()
        
        self.hlp_optim = Adam(self.hlp.parameters(),lr=hypers.lr)
        self.actor_optim = Adam(self.actor.parameters(),lr=hypers.lr)
        self.q_optim = Adam(chain(self.q1.parameters(),self.q2.parameters()),lr=hypers.lr,fused=True)
    
        self.entropy_target = -hypers.action_dim
        self.log_alpha = torch.tensor(1.0,requires_grad=True,device=hypers.device)  
        self.alpha_optim = Adam([self.log_alpha],lr=hypers.lr)
        
        self.storage_path = storage_path
        self.n = 0 # tracking number for model data saving  
            
    
    def save(self, step):
        check = {
            "high level state": self.hlp.state_dict(),
            "actor state": self.actor.state_dict(), 
            "q1 state": self.q1.state_dict(),
            "q1 target": self.q1_target.state_dict(),
            "q2 state": self.q2.state_dict(),
            "q2 target": self.q2_target.state_dict(),
            
            "self.high level optim state": self.hlp_optim.state_dict(),
            "actor optim state" : self.actor.optim.state_dict(),
            "alpha optim state": self.alpha_optim.state_dict(),

            "log_alpha": self.log_alpha,
         }
        torch.save(check,f"{self.storage_path}{step}.pth")

    
    def load(self, model_path=None, strict=True):
        if model_path is not None:
            check = torch.load(model_path, weights_only=False, map_location=hypers.device)
            self.hlp_optim.load_state_dict(check["high level optim state"], strict)
            self.actor.load_state_dict(check["actor state"], strict)
            self.q1.load_state_dict(check["q1 state"],strict)
            self.q1_target.load_state_dict(check["q1 target"], strict)
            self.q2.load_state_dict(check["q2 state"], strict)
            self.q2_target.load_state_dict(check["q2 target"], strict)
            
            self.actor.optim.load_state_dict(check["actor optim state"])
            self.q1_optim.load_state_dict(check["q1 optim state"])
            self.q2_optim.load_state_dict(check["q2 optim state"])

            self.log_alpha.data.copy_(check["log_alpha"].data)
            self.alpha_optim.load_state_dict(check["alpha optim state"])

    
    def log_metrics(self,
        alpha_loss, 
        alpha, 
        alpha_log_action, 
        min_q, new_action, 
        policy_loss, 
        action, 
        alpha_log_nx_action, 
        min_q_target
        ):

        mlflow.log_metrics(
            {
                "Main/entropy loss": alpha_loss.item(),
                "Main/alpha value": alpha.item(),
              
                "policy/log action": alpha_log_action.item(),
                "policy/pred min Q target": min_q.mean().item(),
                "policy/new action variance": new_action.var().item(),
                "policy/loss Policy": policy_loss.item(),
                "policy/action variance": actions.var().item(),

                "critic/alpha log nx action": alpha_log_nx_action.item(),
                "critic/pred min Q target": min_q_target.mean().item(),
            },
            step=step
        )
    
    def pre_launch(self,run_id):
        self.load(model_path=None)
        self.actor_cpu = Actor()
        self.actor_cpu.load_state_dict(self.actor.state_dict()) # important when resuming with a pretrained model
        self.actor_cpu.share_memory()
        
        ep_queue = mp.Queue(maxsize=10) 
        process__ = []
   
        for n in range(5):
            step_thread = mp.Process(target=step, args=(ep_queue,actor_cpu,), daemon=True)
            process__.append(step_thread)
            step_thread.start()

        print_queue_loading(ep_queue)
        
        buffer = create_buffer() # init and share memory of tensors in buffer 
        current_capacity = buffer[-1]
        for tensor in buffer : 
            tensor.share_memory_()

        self.batch_queue = mp.Queue(maxsize=10)
        filler_thread = Thread(target=filler,args=(buffer,ep_queue,run_id,),daemon=True)
        filler_thread.start()

        while not current_capacity.item() == 20:
            print(current_capacity)
            time.sleep(0.2)
        
        sampler_thread = Thread(target=sampler,args=(buffer,batch_queue,),daemon=True)
        sampler_thread.start()


    def train(self):
        mlflow.set_experiment("sac-stack-Robosuite")
        with mlflow.start_run() as run:
            run_id = run.info.run_id

            self.pre_launch(run_id) # !!!

            alpha = self.log_alpha.exp()

            for traj in tqdm(range(hypers.max_steps + 1),total=hypers.max_steps + 1):
                states,nx_states,reward,terminated,actions = self.batch_queue.get()
                # moving data to the gpu
                states = states.to(hypers.device,dtype=torch.float)
                nx_states = nx_states.to(hypers.device,dtype=torch.float)
                reward = reward.to(hypers.device,dtype=torch.float)
                terminated = terminated.to(hypers.device,dtype=torch.float)
                actions = actions.to(hypers.device,dtype=torch.float)

                with torch.no_grad():
                    nx_actions,log_nx_actions,_ = self.actor(nx_states)
                    min_q_target = torch.min(self.q1_target(nx_states,nx_actions),self.q2_target(nx_states,nx_actions))
                    q_target = reward + hypers.gamma * (1-terminated) * (min_q_target - alpha.detach() * log_nx_actions)
                    # R(st|at) + gamma * (Q(st,at) - alpha * log pi(at|st))
                 
                q1_pred = self.q1(states,actions) 
                q2_pred = self.q2(states,actions)

                loss = F.smooth_l1_loss(q1_pred,q_target) + F.smooth_l1_loss(q2_pred,q_target) 
                self.q_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(chain(self.q1.parameters(),self.q2.parameters()),1.0)
                self.q_optim.step() 

                for q1_pars,q1_target_pars in zip(self.q1.parameters(),self.q1_target.parameters()):
                    q1_target_pars.data.mul_(1.0 - hypers.tau).add_(q1_pars.data,alpha=hypers.tau)
            
                for q2_pars,q2_target_pars in zip(self.q2.parameters(),self.q2_target.parameters()):
                    q2_target_pars.data.mul_(1.0 - hypers.tau).add_(q2_pars.data,alpha=hypers.tau)
                
                # freezing critcs weights
                for p in self.q1.parameters() : p.requires_grad = False
                for p in self.q2.parameters() : p.requires_grad = False

                new_action,log_pi,_ = self.actor(states)
                min_q = torch.min(self.q1(states,new_action),self.q2(states,new_action))
                policy_loss = ((alpha.detach() * log_pi) -  min_q).mean() # alpla * log policy(at|st) - Q(st,at)
                
                self.actor.optim.zero_grad(set_to_none=True)
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)
                self.actor.optim.step()

                self.actor_cpu.load_state_dict(self.actor.state_dict()) # update cpu weights

                for p in self.q1.parameters() : p.requires_grad = True
                for p in self.q2.parameters() : p.requires_grad = True

                # Entropy auto tune 
                alpha_loss = (self.log_alpha * (-log_pi - self.entropy_target).detach()).mean()
                self.alpha_optim.zero_grad(set_to_none=True)
                alpha_loss.backward() 
                self.alpha_optim.step()
                alpha = self.log_alpha.exp()

                if traj > 0 and traj % int(20e3) == 0 :
                    self.n+=1
                    self.save(self.n)

                if traj > 0 and traj % int(1e3) == 0 :
                    alpha_log_action = (alpha * log_pi).mean()
                    alpha_log_nx_actions = (alpha * log_nx_actions).mean()

                    self.log_metrics(
                        alpha_loss, 
                        alpha, 
                        alpha_log_action, 
                        min_q, 
                        new_action, 
                        policy_loss, 
                        action, 
                        alpha_log_nx_action, 
                        min_q_target
                    )

                                            

if __name__ == "__main__": 
    mp.set_start_method("spawn",force=True)
    mp.set_sharing_strategy("file_system")
    main(storage_path="./").train(True)
