<!-- ![img](https://github.com/Edision-liu/Reinformancement-learning-on-MAPE/blob/main/videos/simple_formation.gif) -->
<!-- ![img](https://github.com/Edision-liu/Reinformancement-learning-on-MAPE/blob/main/videos/point_to_point.gif) -->
<img src="https://github.com/Edision-liu/Reinformancement-learning-on-MAPE/blob/main/videos/simple_formation.gif" width="350px" height="350px">     <img src="https://github.com/Edision-liu/Reinformancement-learning-on-MAPE/blob/main/videos/point_to_point.gif" width="350px" height="350">

# Swarm intelligence on reinforcement learning for more than 50 agents without collision

## Background
Swarm intelligence has broad research prospects in military, daily life, and multi-role games. It aims at exploring the complicated relationships among multi-agents to stimulate the co-evolution and the emergence of intelligent decision-making, such as collaberation and confrontation scenarios.
Based on MAPE (Multi-agent Particle Environment) and Reinforcement learning, we propose a variable-step learning strategy to facilite the convergence speed of reinforcement learning for more than 50 agents, and design a collision regulation for the agents with the inspiration from the repulsive force field. Experiments prove the efficency of our method and well-trained agents demonstrate the impressive collision-avoidance behavior.


## Installation
See `requirements.txt` file for the list of dependencies. Create a virtualenv with python and setup everything by executing `pip install -r requirements.txt`. 

## Arguments
See `arguments.py` file for the list of various command line arguments one can set while running scripts. 

  `--env-name` for the specific task: simple_spread, simple_formation, simple_line

  `--num-agents` for the specific number of agents to complete the task

  `--render` for whether to visualize the moving process of the agents

  `--test` for the test process

  `--load-dir` for the filedir to load your checkpoint

## Evaluation
`python eval.py`

Trained checkpoints download from: https://pan.baidu.com/s/13At3DIt67NpLx_b_U82oJQ?pwd=fujl

## Normal Training
Training on **Coverage Control** (`simple_spread`) environment can be started by running:

`python main.py`
(Specify the flag `--test` if you do not want to save anything.)

## Curriculum Training
To start curriculum training, specify the number of agents in `automate.py` file and execute:

`python automate.py --env-name simple_spread --entity-mp --save-dir 0`

## Transfer 
You can also continue training from a saved model. For example, for training a team of 5 agents in `simple_spread` task from a policy trained with 3 agents, execute:

`python main.py --env-name simple_spread --entity-mp --continue-training --load-dir models/ss/na3_uc.pt --num-agents 5`


## Other Instruction
_This codeblock are improved on 'Learning Transferable Cooperative Behavior in Multi-Agent Teams', the official repository is available at https://arxiv.org/abs/1906.01202_

## Contact
### For any queries, feel free to raise an issue or contact the authors at lzhihao@tju.edu.cn
