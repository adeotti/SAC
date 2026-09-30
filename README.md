**Lift**  

The Lift task is straightforward: the goal for the robot is to grasp and lift a cube off the table. The robot used for this task, as well as the subsequent Stacking task, is the Franka Panda robot with three fingers (Using three fingers instead of two might take a little longer to train, but the difficulty of the task remains the same.) The vanilla SAC algorithm (Haarnoja et al. 2019) was used for this task and the model should converge around ~1 million gradient steps on batched training data.


**Async design**

For both systems, data collection and training is decoupled to increase training speed and reduce CPU-GPU synchronization overhead. 5 to 10 workers (each holding a shared CPU copy of both the high level and the low level policy) are launched for data collection and they asynchronously push collected episodes to a circular replay buffer, no test was done to study wether the number of worker affect convergence (considering the huge diversity in the transitions collected) but the size of the buffer seem to directly affect learning. The circular buffer holds a maximum of 100 episodes and the training loop of the high and low level policy runs in different threads and on different stream on the GPU (a single GPU was used for this experiment). This Async design is directly derived from the method seen in (Mninh et al., 2016; Nair et al., 2015) but unlike in (Mninh et al., 2016), the update of the CPU copies is done at each gradient step after the update of the GPU version of each policy (high level and low level) and No gradient accumulation was done. Adam was used as optimizer.

The hyperparameters for the Lift task can be seen in the head of the `lift.py` file and for the stack envrionment, in the `configs.py` inside the `stack_env` repo. The training data for each environment is provided in each repo as an `mlflow db` file. Syntax to run mlflow (example with lift repo) : `cd lift_env && mlflow server`. Training should also be run using python optimize so all assertion could be ignored, launching training using `python -O stack.py`


**References**  

Addressing Function Approximation Error in Actor-Critic Methods; Scott Fujimoto, Herke van Hoof, David Meger (ICML 2018).  

Soft Actor-Critic Algorithms and Applications; Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine (arXiv 2018/2019).  

Data-Efficient Hierarchical Reinforcement Learning; Ofir Nachum, Shixiang Gu, Honglak Lee, Sergey Levine (NeurIPS 2018).  

Asynchronous Methods for Deep Reinforcement Learning; Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu (ICML 2016).  

Massively Parallel Methods for Deep Reinforcement Learning; Arun Nair, Praveen Srinivasan, Sam Blackwell, Cagdas Alcicek, Rory Fearon, Alessandro De Maria, Vedavyas Panneershelvam, Mustafa Suleyman, Charles Beattie, Stig Petersen, Shane Legg, Volodymyr Mnih, Koray Kavukcuoglu, David Silver (arXiv 2015).  

[haarnoja/sac](https://github.com/haarnoja/sac) Official TensorFlow implementation by the authors  

[OpenAI Spinning Up (PyTorch)](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/sac) OpenAI reference implementation

[pranz24/pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic) PyTorch SAC implementation

[denisyarats/pytorch_sac](https://github.com/denisyarats/pytorch_sac) PyTorch SAC benchmark implementation

[ARISE-Initiative/robosuite-benchmark](https://github.com/ARISE-Initiative/robosuite-benchmark) Robosuite benchmarking repository

[Robosuite Lift a Stack doc page](https://robosuite.ai/docs/modules/environments.html#task-descriptions)

