from tqdm import tqdm
import torch
from configs import hypers


__all__ = ["create_storage", "create_buffer"]


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


def step_envs(queue,policy,goal): # episode collection method  
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
                action,_,_ = policy(torch.as_tensor(obs)) # TODO pass the goal too 
                action = action.squeeze()
            
            nx_state,reward,done,trunc,info = env.step(action.tolist())
            # TODO extract the achived goal and replace reward to be goal specific
            
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


