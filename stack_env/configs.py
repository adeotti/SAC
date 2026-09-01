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
    hl_action_dim = 6  # high level action dim
    batch_size = 1024
    policy_lr = .0001 
    critic_lr = .001
    alpha_lr = 3e-4
    gamma = .99
    tau = .005
    warmup = 2000
    max_hlp_update_steps = int(10e5) 
    max_llp_update_steps = int(10e6)
    num_envs = 10
    horizon = 500
    buffer_size = 400  # 400*horizon = 200k steps
    num_rollout_workers = 6
    low_queue_maxsize = 10 # max size of every queue
    high_queue_maxsize = 20
    buffer_min_capacity = 20 # min buffer capacity before starting sampling
    high_level_train_steps = 10 # number of update on the policy, critics, targets and alpha during high level training 

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
