import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.common import Autoreset

import torch
import mlflow
import numpy as np
from configs import *
from tqdm import tqdm

__all__ = ["step_envs", "create_buffer", "filler"]


def vec_env(): # environment creation 
    def make_env():
        x = robosuite.make(env_name="Stack", **env_configs) 
        x = GymWrapper(x,list(x.observation_spec()))
        x.metadata = {"render_mode":[]}
        x = Autoreset(x)
        return x 
    env = SyncVectorEnv([make_env for _ in range(hypers.num_envs)])
    return env


def get_local_reward(env,hl_goal): # TODO : review math and optimize this slow code 
    mujoco_layer = env.unwrapped
    data_list = []
    for env in mujoco_layer.envs:
        cubeA_pos = env.unwrapped._get_observations()["cubeA_pos"]
        gripper_to_cubeA = env.unwrapped._get_observations()["gripper_to_cubeA"]
        stack = torch.cat([torch.from_numpy(cubeA_pos),torch.from_numpy(gripper_to_cubeA)])
        data_list.append(stack) 

    obs_goal = torch.stack(data_list,dim=0).float() # each env data are on the x axis
    diff = torch.linalg.norm((hl_goal-obs_goal),dim=-1)
    return -diff.mean(), obs_goal


def create_storage(): # for episodic data storage
    obs_dim = (hypers.num_envs,hypers.obs_dim)     
    act_dim = (hypers.num_envs,hypers.ll_action_dim)
    return (
        torch.empty((hypers.horizon, *obs_dim), dtype=torch.half),  # states
        torch.empty((hypers.horizon, *obs_dim), dtype=torch.half),  # nx states
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.half),  # environment reward
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.half),  # local reward (L2 norm)
        torch.empty((hypers.horizon, hypers.num_envs,), dtype=torch.bool),  # done
        torch.empty((hypers.horizon, *act_dim), dtype=torch.half),  # actions
        torch.empty((hypers.horizon, hypers.num_envs, 6), dtype=torch.half),  # hl goal 
        torch.empty((hypers.horizon, hypers.num_envs, 6), dtype=torch.half)  # observed goal
    )


def create_buffer(): # circular buffer 
    n_batch = torch.tensor(hypers.buffer_size) 
    obs_dim = (hypers.num_envs, hypers.obs_dim)
    act_dim = (hypers.num_envs, hypers.ll_action_dim)
    goal_dim = (hypers.num_envs, hypers.hl_action_dim)
    return (
        torch.zeros((n_batch, hypers.horizon, *obs_dim), dtype=torch.half),  # states
        torch.zeros((n_batch, hypers.horizon, *obs_dim), dtype=torch.half),  # nx states
        torch.zeros((n_batch, hypers.horizon, hypers.num_envs), dtype=torch.half),  # environment rewards
        torch.zeros((n_batch, hypers.horizon, hypers.num_envs), dtype=torch.half),  # local rewards
        torch.zeros((n_batch, hypers.horizon, hypers.num_envs,), dtype=torch.bool),  # done
        torch.zeros((n_batch, hypers.horizon, *act_dim), dtype=torch.half),  # actions
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

        while True:
            goal,_,_ = hlp(torch.as_tensor(obs)) # sample goal with high level policy 

            for n in range(hypers.horizon):
                if global_step < hypers.warmup: 
                    action = env.action_space.sample()
                else:
                    action,_,_ = llp(torch.as_tensor(obs),goal) 
                    action = action.squeeze()
                
                nx_state,env_reward,done,trunc,info = env.step(action.tolist())
                local_reward,obs_goal = get_local_reward(env, goal) # TODO review formulas and h function
                        
                stor_curr_states[n].copy_(torch.as_tensor(obs))
                stor_nx_states[n].copy_(torch.as_tensor(nx_state))
                stor_rewards[n].copy_(torch.from_numpy(env_reward))
                stor_local_rewards[n].copy_(local_reward)
                stor_dones[n].copy_(torch.from_numpy(done))
                stor_actions[n].copy_(torch.as_tensor(action))
                stor_hl_goals[n].copy_(goal)
                stor_obs_goals[n].copy_(obs_goal)

                obs = nx_state
                global_step += 1

                if n > 0 and n % hypers.horizon-1 == 0:
                    data = (
                        stor_curr_states.clone(),
                        stor_nx_states.clone(),
                        stor_rewards.clone(),
                        stor_local_rewards.clone(),
                        stor_dones.clone(),
                        stor_actions.clone(),
                        stor_hl_goals.clone(),
                        stor_obs_goals.clone()
                    )
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


def print_queue_loading(queue): # tracking queue size mainly during warmup phase
    pbar = tqdm(total=queue._maxsize,desc="Warmup")
    while True:
        pbar.n = queue.qsize()
        pbar.refresh()
        time.sleep(0.1)
        if queue.qsize() == queue._maxsize:
            break
    pbar.n = queue.qsize()
    pbar.refresh()
    pbar.close()


