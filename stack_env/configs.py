import torch
from robosuite import load_composite_controller_config
from dataclasses import dataclass

__all__ = ["hypers", "env_configs"]

@dataclass(frozen=True)
class Hypers:
    ROBOT = "Panda"
    env_name = None
    device = torch.device("cuda:0")
    obs_dim = 81   
    ll_action_dim = 9  # low level action dim
    hl_action_dim = 3  # high level action dim
    batch_size = 1024
    policy_lr = 1e-4 
    critic_lr = 1e-3
    alpha_lr = 1e-4
    gamma = .99
    tau = 5e-3
    max_hlp_update_steps = int(10e6) 
    max_llp_update_steps = int(10e6)
    num_envs = 10
    horizon = 500
    buffer_size = 100  # 100*horizon = 50k steps
    num_rollout_workers = 10
    warmup = 20_000 // (num_rollout_workers*num_envs)
    low_queue_maxsize = 10 
    high_queue_maxsize = 10
    buffer_min_capacity = 20 # min buffer capacity before starting sampling
    c = 10 # frequency of the high level policy 

hypers = Hypers()

cont_config = controller = load_composite_controller_config(robot=hypers.ROBOT)
env_configs = {
    "robots": "Panda",
    "controller_configs": cont_config,
    "gripper_types": "JacoThreeFingerDexterousGripper",
    "has_renderer": False,
    "use_camera_obs": False,
    "has_offscreen_renderer": False,
    "reward_shaping": True,  # Dense rewards env version 
    "horizon": hypers.horizon,  # Max steps before reset or trunc = True
    "control_freq": 20,
    "reward_scale": 1.0
}
