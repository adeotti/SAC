import warnings,logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import torch,sys,robosuite
from lift import Actor
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.wrappers.stateful_observation import NormalizeObservation
from robosuite import load_composite_controller_config

controller = load_composite_controller_config(robot="Panda")
env = robosuite.make(
    controller_configs = controller,
    env_name="Lift", 
    robots="Panda",  
    gripper_types="JacoThreeFingerDexterousGripper",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon = 500,
    control_freq = 20
)
env = GymWrapper(env,list(env.observation_spec()))
#env = NormalizeObservation(env)

obs = env.reset()[0]
policy = Actor()

checkpoint = torch.load("./39.pth",map_location="cpu",weights_only=False) 
policy.load_state_dict(checkpoint["actor state"])

""" !!! when using that NormalizeObservation wrapper
env.obs_rms.mean = checkpoint["obs_rms_mean"].numpy()
env.obs_rms.var = checkpoint["obs_rms_var"].numpy()
env.update_running_mean = True 
"""
for i in range(500*30):
    _,_,action = policy(torch.from_numpy(obs).float())
    obs,reward,done,trunc,info = env.step(action.detach().numpy())
    env.render()
    if trunc or done:
        obs = env.reset()[0]
