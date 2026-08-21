import warnings,logging
warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)

import torch
import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_composite_controller_config

controller = load_composite_controller_config(robot="Panda")
env = robosuite.make(
    controller_configs = controller,
    env_name="Stack", 
    robots="Panda",  
    gripper_types="JacoThreeFingerDexterousGripper",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon = 500,
    control_freq = 20
)
env = GymWrapper(env,list(env.observation_spec()))

