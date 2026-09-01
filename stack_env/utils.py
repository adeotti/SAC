import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.common import Autoreset

import torch
import mlflow
import time
import numpy as np
from configs import *
from tqdm import tqdm


__all__ = [
    "step_envs",
    "create_buffer",
    "generate_random_samples",
    "filler", 
    "low_level_sampler",
    "high_level_sampler",
    "print_queue_loading"
]


CUBEA_SLICE = slice(0, 3)  # "cubeA_pos"
GRIPPER_TO_CUBEA_SLICE = slice(17, 20)  # "gripper_to_cubeA"

def vec_env(): # environment creation 
    def make_env():
        x = robosuite.make(env_name="Stack", **env_configs) 
        x = GymWrapper(x, sorted(list(x.active_observables)))
        x.metadata = {"render_mode":[]}
        x = Autoreset(x)
        return x 
    return SyncVectorEnv([make_env for _ in range(hypers.num_envs)])


def get_local_reward(state, next_state, hl_goal, max_dist=1.41):
    cube_pos_hl_goal = hl_goal[:, :3]
    target_cubea = state[:, CUBEA_SLICE] + cube_pos_hl_goal
    nx_cubea = next_state[:, CUBEA_SLICE]
    cube_l = torch.linalg.norm((target_cubea - nx_cubea), dim=-1, keepdim=True)
    
    gripper_hl_goal = hl_goal[:, 3:]
    target_gripper = state[:, GRIPPER_TO_CUBEA_SLICE] + gripper_hl_goal
    nx_gripper = next_state[:, GRIPPER_TO_CUBEA_SLICE]
    gripper_l = torch.linalg.norm((target_gripper - nx_gripper), dim=-1, keepdim=True)
    
    reward = -(cube_l+gripper_l)
    observed_goal = torch.cat([nx_cubea, nx_gripper], dim=-1) 
    return reward.squeeze(), observed_goal


def generate_random_samples(N=32): # random samples for testing 
    s_states = torch.randn((N, hypers.obs_dim), dtype=torch.half)
    s_nx_state = torch.randn((N, hypers.obs_dim), dtype=torch.half)
    s_reward = torch.randn((N, 1), dtype=torch.half)
    s_local_rewards = torch.randn((N, 1), dtype=torch.half)
    s_dones = torch.randint(0, 2, (N, 1), dtype=torch.bool)
    s_actions = torch.randn((N, hypers.ll_action_dim), dtype=torch.half)
    s_hl_goals = torch.randn((N, hypers.hl_action_dim), dtype=torch.half)
    s_obs_goals = torch.randn((N, hypers.hl_action_dim), dtype=torch.half)
    return s_states, s_nx_state, s_reward, s_local_rewards, s_dones, s_actions, s_hl_goals, s_obs_goals


def create_storage(): # for episodic data storage
    obs_dim = (hypers.num_envs,hypers.obs_dim)     
    act_dim = (hypers.num_envs,hypers.ll_action_dim)
    return (
        torch.empty((hypers.horizon, *obs_dim), dtype=torch.half),   # states
        torch.empty((hypers.horizon, *obs_dim), dtype=torch.half),   # nx states
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.half),   # environment reward
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.half),   # local reward (L2 norm)
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.bool),   # done
        torch.empty((hypers.horizon, *act_dim), dtype=torch.half),           # actions
        torch.empty((hypers.horizon, hypers.num_envs, 6), dtype=torch.half), # hl goal 
        torch.empty((hypers.horizon, hypers.num_envs, 6), dtype=torch.half)  # observed goal
    )


def create_buffer(): # circular buffer 
    n_batch = torch.tensor(hypers.buffer_size) 
    obs_dim = (hypers.num_envs, hypers.obs_dim)
    act_dim = (hypers.num_envs, hypers.ll_action_dim)
    goal_dim = (hypers.num_envs, hypers.hl_action_dim)
    return (
        torch.zeros((n_batch, hypers.horizon, *obs_dim), dtype=torch.half),         # states
        torch.zeros((n_batch, hypers.horizon, *obs_dim), dtype=torch.half),         # nx states
        torch.zeros((n_batch, hypers.horizon, hypers.num_envs), dtype=torch.half),  # environment rewards
        torch.zeros((n_batch, hypers.horizon, hypers.num_envs), dtype=torch.half),  # local rewards
        torch.zeros((n_batch, hypers.horizon, hypers.num_envs,), dtype=torch.bool), # done
        torch.zeros((n_batch, hypers.horizon, *act_dim), dtype=torch.half),   # actions
        torch.zeros((n_batch, hypers.horizon, *goal_dim), dtype=torch.half),  # hl goals
        torch.zeros((n_batch, hypers.horizon, *goal_dim), dtype=torch.half),  # observed goals
        torch.tensor(0)  # current buffer capacity
    ) 


def step_envs(queue, hlp, llp): # episode collection method  
    with torch.no_grad():
        stor_curr_states, stor_nx_states, stor_rewards, stor_local_rewards, stor_dones, stor_actions, stor_hl_goals, stor_obs_goals = create_storage()
    
        env = vec_env()
        pointer = 0
        global_step = 0
        obs = torch.from_numpy(env.reset()[0])
        c = 10
        while True:
            for n in range(hypers.horizon):
                obs = torch.as_tensor(obs)

                if n % c == 0: goal = hlp(obs)[0] # sample goal with high level policy

                if global_step < hypers.warmup: 
                    action = env.action_space.sample()
                else:
                    action = llp(obs, goal)[0] 
                    action = action.squeeze()
                
                nx_state,env_reward,done,trunc,info = env.step(action.tolist())
                local_reward,obs_goal = get_local_reward(obs, torch.as_tensor(nx_state), goal)
                      
                stor_curr_states[n].copy_(obs)
                stor_nx_states[n].copy_(torch.as_tensor(nx_state))
                stor_rewards[n].copy_(torch.from_numpy(env_reward))
                stor_local_rewards[n].copy_(local_reward)
                stor_dones[n].copy_(torch.from_numpy(done))
                stor_actions[n].copy_(torch.as_tensor(action))
                stor_hl_goals[n].copy_(goal)
                stor_obs_goals[n].copy_(obs_goal)

                obs = nx_state
                global_step += 1

                if n == hypers.horizon-1:
                    data = (
                        stor_curr_states, stor_nx_states, stor_rewards, stor_local_rewards, stor_dones, stor_actions, stor_hl_goals, stor_obs_goals
                    )
                    data = [tensor.clone() for tensor in data]
                    queue.put(data)
                 

def filler(buffer, episodes_queue, mlflow_run_id): # method used by a wroker to fill the buffer 
    b_state, b_nx_state, b_rewards, b_local_rewards, b_dones, b_actions, b_hl_goals, b_obs_goals, current_capacity = buffer
    n_batch = hypers.buffer_size
    global_idx = 0

    while True: 
        ep_curr_states, ep_nx_states, ep_rewards, ep_local_rewards, ep_dones, ep_actions, ep_hl_goals, ep_obs_goals = episodes_queue.get()

        p = global_idx % n_batch

        b_state[p].copy_(ep_curr_states)
        b_nx_state[p].copy_(ep_nx_states)
        b_rewards[p].copy_(ep_rewards)
        b_local_rewards[p].copy_(ep_local_rewards)
        b_dones[p].copy_(ep_dones)
        b_actions[p].copy_(ep_actions)
        b_hl_goals[p].copy_(ep_hl_goals)
        b_obs_goals[p].copy_(ep_obs_goals)
        
        global_idx += 1
        current_capacity.copy_(torch.tensor(min(global_idx, n_batch)))
        
        mean_return = ep_rewards.sum().item() / hypers.num_envs # tracking rewards per episodes
        mlflow.log_metric("Main/mean reward", mean_return,run_id=mlflow_run_id, step=global_idx) 
        

def low_level_sampler(buffer,low_gpu_stream): # method used by a worker to sample from the buffer then put the sample in a queue
    b_state, b_nx_state, b_rewards, b_local_rewards, b_dones, b_actions, b_hl_goals, b_obs_goals, current_capacity = buffer
 
    while True:
        if current_capacity.item() < hypers.buffer_min_capacity:
            time.sleep(0.1)
            continue
        
        idx_chunks = torch.randint(0,current_capacity,(hypers.batch_size,))
        idx_horizons = torch.randint(0,hypers.horizon,(hypers.batch_size,))
        idx_envs = torch.randint(0,hypers.num_envs,(hypers.batch_size,))

        s_states = b_state[idx_chunks, idx_horizons, idx_envs]                               # [1024, 162]
        s_nx_state = b_nx_state[idx_chunks, idx_horizons, idx_envs]                          # [1024, 162]
        s_reward = b_rewards[idx_chunks, idx_horizons, idx_envs].unsqueeze(-1)               # [1024, 1]
        s_local_rewards = b_local_rewards[idx_chunks, idx_horizons, idx_envs].unsqueeze(-1)  # [1024, 1]
        s_dones = b_dones[idx_chunks, idx_horizons, idx_envs].unsqueeze(-1)  # [1024, 1]
        s_actions = b_actions[idx_chunks, idx_horizons, idx_envs]            # [1024, 9]
        s_hl_goals = b_hl_goals[idx_chunks, idx_horizons, idx_envs]          # [1024, 6]
        s_obs_goals = b_obs_goals[idx_chunks, idx_horizons, idx_envs]        # [1024, 6]

        data = (s_states, s_nx_state, s_local_rewards, s_dones, s_actions, s_hl_goals, s_obs_goals)
        low_gpu_stream.put(data, block=True)


def high_level_sampler(buffer,high_gpu_stream):
    data = [(tensor.unsqueeze(-1) if tensor.dim() < 4 else tensor) for tensor in buffer]
    b_states, b_nx_states, b_rewards, _, b_dones, b_actions, b_hl_goals, b_obs_goals, current_capacity = data
    
    while True:
        if current_capacity.item() < hypers.buffer_min_capacity:
            time.sleep(0.1)
            continue
        
        batch_idx = torch.randint(0, current_capacity, (1024,1))                              # [1024, 1]
        horizon_idx = torch.randint(0, (500-10+1), (1024, 1)) + torch.arange(10).unsqueeze(0) # [1024, 1 ] + [1, 10]  -->  [1024, 10]
        env_idx = torch.randint(0, 10, (1024,1))                                              # [1024, 1]

        # extracting sequence samples 
        s_states   = b_states[batch_idx, horizon_idx, env_idx]     # [1024, 10, 81]
        s_nx_states = b_nx_states[batch_idx, horizon_idx, env_idx] # [1024, 10, 81]
        s_rewards = b_rewards[batch_idx, horizon_idx, env_idx]     # [1024, 10, 1] 
        s_dones = b_dones[batch_idx, horizon_idx, env_idx]         # [1024, 10, 1]
        s_actions  = b_actions[batch_idx, horizon_idx, env_idx]    # [1024, 10, 9]
        s_hl_goals = b_hl_goals[batch_idx, horizon_idx, env_idx]   # [1024, 10, 6]
        s_obs_goals = b_obs_goals[batch_idx, horizon_idx, env_idx] # [1024, 10, 6]
        #-
        s_nx_states = s_nx_states[:, -1, :]                    # [1024, 81]
        s_rewards = torch.sum(s_rewards, dim=1, keepdim=True)  # [1024, 1, 1], summing all reward in the sequence 
        s_dones = s_dones[:, -1, :]                            # [1024, 1]
        s_hl_goals = s_hl_goals[:, 0, :]                       # [1024, 6]
        s_obs_goals = s_obs_goals[:, -1, :]                    # [1024, 6]
        
        data = (s_states, s_nx_states, s_rewards, s_dones, s_actions, s_hl_goals, s_obs_goals)
        high_gpu_stream.put(data, block=True)
        

def print_queue_loading(queue, name, break_point): # tracking queue size 
    pbar = tqdm(total=queue._maxsize, desc=name)
    while True:
        pbar.n = queue.qsize()
        pbar.refresh()
        time.sleep(0.1)
        if queue.qsize() == break_point:
            break
    pbar.n = queue.qsize()
    pbar.refresh()
    pbar.close()


