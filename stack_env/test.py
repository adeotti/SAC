import warnings,logging
warnings.filterwarnings("ignore") ; logging.disable(logging.CRITICAL)

import torch
import robosuite
from robosuite.wrappers.gym_wrapper import GymWrapper
from robosuite import load_composite_controller_config
from networks import HLP, LLP

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
env = GymWrapper(env,sorted(list(env.active_observables)))
obs = torch.from_numpy(env.reset()[0]).float()
hl_policy = HLP()
ll_policy = LLP()

checkpoint = torch.load("./69.pth",map_location="cpu",weights_only=False) 
hl_policy.load_state_dict(checkpoint["high level state"])
ll_policy.load_state_dict(checkpoint["low level state"])

for i in range(500*30):
    if i%10==0:
        goal = hl_policy(torch.as_tensor(obs).float())[-1]
    
    action = ll_policy(torch.as_tensor(obs).float(), goal)[-1]
    obs,reward,done,trunc,info = env.step(action.detach().numpy())
    env.render()
    if trunc or done:
        obs = env.reset()[0]
