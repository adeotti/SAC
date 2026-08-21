import torch
from robosuite import load_composite_controller_config
from dataclasses import dataclass

__all__ = ["hypers", "env_configs"]

@dataclass(frozen=True)
class Hypers:
    ROBOT = "Panda"
    env_name = None
    device = torch.device("cuda:0")
    obs_dim = 162   
    ll_action_dim = 9  # low level action dim
    hl_action_dim = 6  # high level action dim
    batch_size = 1024
    policy_lr = .0001 
    critic_lr = .001
    alpha_lr = 3e-4
    gamma = .99
    tau = .005
    warmup = 2000
    max_steps = int(10e6)
    num_envs = 10
    horizon = 500
    buffer_size = 400  # 400*horizon = 200k steps

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
