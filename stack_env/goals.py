# modified code from the staged_rewards method in the source code of the Stack environment
import torch,sys
import numpy as np

def goal_reach_block(env): # reach
    obs = env.unwrapped._get_observations()
    cubeA_pos = obs.get("cubeA_pos")
    cubeB_pos = obs.get("cubeB_pos")
    gripper_to_cubeA = np.linalg.norm(obs.get("gripper_to_cubeA")) # scalar distance 
    
    reach_score = 1.0 - np.tanh(10.0 * gripper_to_cubeA)
    
    if reach_score > 0.95:
        achieved = torch.tensor([1.0], dtype=torch.float)
    else:
        achieved = torch.tensor([0.0], dtype=torch.float)

    reach_reward = reach_score
    goal_obs = achieved  # target goal [1.0,]
    return reward_reach,goal_obs
    

def goal_lift_block(env): # grasp and lift
    mujoco_wrapper = env.unwrapped
    gripper_type = mujoco_wrapper.robot_configs[0].get("gripper_type")
    grasped = mujoco_wrapper._check_grasp(gripper=gripper_type, object_geoms=mujoco_wrapper.cubeA)
    grasp_obs_tensor = torch.tensor([grasped], dtype=torch.float)
    reward_lift = 1 if grasped else 0.0
    
    cubeA_height = mujoco_wrapper._get_observations().get("cubeA_pos")[2]
    table_height = mujoco_wrapper.table_offset[2]
    cubeA_lifted = cubeA_height > table_height + 0.04  
    cubeA_lifted_tensor = torch.tensor([cubeA_lifted], dtype=torch.float) 

    goal_obs = torch.cat([grasp_obs_tensor, cubeA_lifted_tensor]) # target goal [1.0, 1.0]
    reward_lift += 1.0 if cubeA_lifted else 0.0

    return reward_lift, goal_obs

   
def goal_stack_blocks(env): # align and stack
    mujoco_wrapper = env.unwrapped
    contact = mujoco_wrapper.check_contact(mujoco_wrapper.cubeA, mujoco_wrapper.cubeB)
    gripper_type = mujoco_wrapper.robot_configs[0].get("gripper_type")
    grasped = mujoco_wrapper._check_grasp(gripper=gripper_type, object_geoms=mujoco_wrapper.cubeA)

    contact_tensor = torch.tensor([contact], dtype=torch.float)
    grasping_tensor = torch.tensor([grasped], dtype=torch.float)
    goal_obs = torch.cat([contact_tensor,gripper_type]) #  target goal [1.0, 0.0]
    if contact_tensor.items() and not grasping_tensor:
        reward_stack = 2.0

    return reward_stack, goal_obs
    
