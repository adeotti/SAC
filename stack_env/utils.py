from tqdm import tqdm

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


