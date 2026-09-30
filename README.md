**Lift**  

The Lift task is straightforward: the goal for the robot is to grasp and lift a cube off the table. The robot used for this task, as well as the subsequent Stacking task, is the Franka Panda robot with three fingers (Using three fingers instead of two might take a little longer to train, but the difficulty of the task remains the same.) The vanilla SAC algorithm (Haarnoja et al. 2019) was used for this task and the model should converge around ~1 million gradient steps on batched training data. 
